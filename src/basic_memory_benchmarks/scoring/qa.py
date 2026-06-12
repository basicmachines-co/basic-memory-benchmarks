"""End-to-end QA scoring: answer generation over retrieved context, then judging.

This is the stage that produces comparable "memory benchmark accuracy" numbers.
Retrieval metrics (recall/MRR) measure the search layer; QA accuracy measures
whether an LLM holding only the retrieved memories can actually answer the
question. Both prompts below are fixed and ship with the repo so any published
number can be audited.
"""

from __future__ import annotations

import json
import re
from concurrent.futures import ThreadPoolExecutor

from basic_memory_benchmarks.llm.runners import LLMRunner, LLMRunnerError
from basic_memory_benchmarks.models import (
    PerQueryRetrievalResult,
    QACaseResult,
    QACategoryMetrics,
    QASummary,
    SearchHit,
)

# Context assembly budget. Multi-fact answers (LoCoMo multi_hop, LongMemEval
# multi-session) need material from several hits; the previous top-5
# matched-chunk join (~1K chars) capped every provider near zero on those
# categories despite ~0.8 retrieval recall.
CONTEXT_MAX_HITS = 10
CONTEXT_MAX_CHARS = 12_000
CONTEXT_CHARS_PER_HIT = 2_500

# The exact abstention sentinel the answer prompt requests. Judged correct only
# when the gold answer itself indicates the question is unanswerable (e.g.
# LoCoMo adversarial cases).
ABSTAIN_SENTINEL = "I don't know"

ANSWER_PROMPT_TEMPLATE = """\
You are answering a question using only the retrieved memories below. The memories
come from past conversations and notes; they may be incomplete.

Question: {question}

Retrieved memories:
{context}

Instructions:
- Answer concisely using only facts found in the retrieved memories.
- If the memories do not contain the information needed to answer, reply with
  exactly: {abstain}
- Do not use outside knowledge. Do not explain your reasoning.

Answer:"""

JUDGE_PROMPT_TEMPLATE = """\
You are grading a question-answering system. Compare the candidate answer to the
gold answer.

Question: {question}
Gold answer: {gold}
Candidate answer: {candidate}

Grading rules:
- correct = true only if the candidate states the same core fact(s) as the gold
  answer. Paraphrase, formatting, and extra correct detail are fine.
- If the gold answer indicates the information is not available (e.g. "not
  mentioned", "no information"), the candidate is correct only if it also
  declines to answer (e.g. "{abstain}").
- A candidate that declines to answer when the gold answer contains a real fact
  is incorrect.
- Partial answers missing a key fact are incorrect.

Reply with only a JSON object: {{"correct": true or false, "reason": "<one sentence>"}}"""

_JSON_OBJECT_PATTERN = re.compile(r"\{.*\}", re.DOTALL)


def build_answer_prompt(question: str, context: str) -> str:
    return ANSWER_PROMPT_TEMPLATE.format(
        question=question,
        context=context if context.strip() else "(no memories were retrieved)",
        abstain=ABSTAIN_SENTINEL,
    )


def build_judge_prompt(question: str, gold: str, candidate: str) -> str:
    return JUDGE_PROMPT_TEMPLATE.format(
        question=question,
        gold=gold,
        candidate=candidate,
        abstain=ABSTAIN_SENTINEL,
    )


def parse_judge_verdict(raw: str) -> tuple[bool, str]:
    """Extract a (correct, reason) verdict from judge output.

    Judges occasionally wrap JSON in prose or code fences; take the first JSON
    object found. A malformed verdict raises so the case is recorded as an
    explicit error rather than silently scored.
    """
    match = _JSON_OBJECT_PATTERN.search(raw)
    if not match:
        raise ValueError(f"Judge returned no JSON object: {raw[:200]}")
    payload = json.loads(match.group(0))
    if not isinstance(payload.get("correct"), bool):
        raise ValueError(f"Judge JSON missing boolean 'correct': {raw[:200]}")
    return payload["correct"], str(payload.get("reason") or "")


def _is_abstention(answer: str) -> bool:
    normalized = answer.strip().strip(".").lower()
    return normalized == ABSTAIN_SENTINEL.strip(".").lower()


def assemble_context(hits: list[SearchHit]) -> str:
    """Build answering context from ranked hits under a character budget.

    Hits are taken in rank order; each contributes up to CONTEXT_CHARS_PER_HIT
    characters and the total is capped at CONTEXT_MAX_CHARS. Sections are
    numbered with their source doc so the answerer can ground multi-fact
    answers across memories. Identical assembly for every provider.
    """
    sections: list[str] = []
    used = 0
    for rank, hit in enumerate(hits[:CONTEXT_MAX_HITS], start=1):
        text = (hit.text or "").strip()
        if not text:
            continue
        snippet = text[:CONTEXT_CHARS_PER_HIT]
        if used + len(snippet) > CONTEXT_MAX_CHARS:
            snippet = snippet[: CONTEXT_MAX_CHARS - used]
            if not snippet:
                break
        source = hit.source_doc_id or hit.source_path or "unknown"
        sections.append(f"[Memory {rank} | source: {source}]\n{snippet}")
        used += len(snippet)
        if used >= CONTEXT_MAX_CHARS:
            break
    return "\n\n".join(sections)


def _row_context(row: PerQueryRetrievalResult) -> str:
    # Prefer assembling from stored hits (richer, budget-controlled); fall
    # back to the legacy pre-joined context for old artifacts without hits.
    if row.hits:
        assembled = assemble_context(row.hits)
        if assembled:
            return assembled
    return row.retrieved_context


def _question_display(row: PerQueryRetrievalResult) -> str:
    """Render the question with its ask-date when the dataset provides one.

    Temporal-reasoning questions ("how many weeks ago...") are unanswerable
    without the reference date, and both the answerer and the judge need the
    same framing.
    """
    question_date = row.metadata.get("question_date")
    if question_date:
        return f"{row.query_text} (question asked on {question_date})"
    return row.query_text


def _score_case(
    row: PerQueryRetrievalResult,
    provider: str,
    answerer: LLMRunner,
    judge: LLMRunner,
) -> QACaseResult:
    question = _question_display(row)
    try:
        answer_result = answerer.complete(build_answer_prompt(question, _row_context(row)))
        judge_result = judge.complete(
            build_judge_prompt(question, row.expected_answer or "", answer_result.text)
        )
        correct, reason = parse_judge_verdict(judge_result.text)
        return QACaseResult(
            provider=provider,
            query_id=row.query_id,
            category=row.category,
            question=row.query_text,
            expected_answer=row.expected_answer or "",
            generated_answer=answer_result.text,
            abstained=_is_abstention(answer_result.text),
            correct=correct,
            judge_reason=reason,
            answer_model=answerer.spec,
            judge_model=judge.spec,
            answer_latency_ms=answer_result.latency_ms,
            answer_input_tokens=answer_result.input_tokens,
            answer_output_tokens=answer_result.output_tokens,
        )
    except (LLMRunnerError, ValueError, json.JSONDecodeError) as exc:
        return QACaseResult(
            provider=provider,
            query_id=row.query_id,
            category=row.category,
            question=row.query_text,
            expected_answer=row.expected_answer or "",
            generated_answer="",
            abstained=False,
            correct=False,
            judge_reason="",
            answer_model=answerer.spec,
            judge_model=judge.spec,
            answer_latency_ms=0.0,
            answer_input_tokens=0,
            answer_output_tokens=0,
            error=str(exc),
        )


def run_qa(
    rows: list[PerQueryRetrievalResult],
    *,
    provider: str,
    answerer: LLMRunner,
    judge: LLMRunner,
    max_workers: int = 4,
) -> tuple[list[QACaseResult], QASummary]:
    """Answer and judge every row that carries an expected answer."""
    scorable = [row for row in rows if row.expected_answer]
    if not scorable:
        return [], QASummary(
            provider=provider,
            answer_model=answerer.spec,
            judge_model=judge.spec,
            total_cases=0,
            correct_count=0,
            error_count=0,
            abstain_count=0,
            accuracy=0.0,
            skipped_reason="No expected answers available for QA scoring",
        )

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        case_results = list(
            pool.map(lambda row: _score_case(row, provider, answerer, judge), scorable)
        )

    by_category: dict[str, QACategoryMetrics] = {}
    for case in case_results:
        bucket = by_category.setdefault(case.category, QACategoryMetrics())
        bucket.total += 1
        bucket.correct += 1 if case.correct else 0
    for bucket in by_category.values():
        bucket.accuracy = bucket.correct / bucket.total if bucket.total else 0.0

    correct_count = sum(1 for case in case_results if case.correct)
    error_count = sum(1 for case in case_results if case.error)
    summary = QASummary(
        provider=provider,
        answer_model=answerer.spec,
        judge_model=judge.spec,
        total_cases=len(case_results),
        correct_count=correct_count,
        error_count=error_count,
        abstain_count=sum(1 for case in case_results if case.abstained),
        accuracy=correct_count / len(case_results),
        by_category=by_category,
        mean_answer_latency_ms=(
            sum(case.answer_latency_ms for case in case_results) / len(case_results)
        ),
        total_answer_input_tokens=sum(case.answer_input_tokens for case in case_results),
        total_answer_output_tokens=sum(case.answer_output_tokens for case in case_results),
    )
    return case_results, summary
