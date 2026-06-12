"""Tests for the end-to-end QA scoring stage."""

from __future__ import annotations

import json

import pytest

from basic_memory_benchmarks.llm.runners import LLMResult, LLMRunner, LLMRunnerError
from basic_memory_benchmarks.models import PerQueryRetrievalResult
from basic_memory_benchmarks.scoring.qa import (
    ABSTAIN_SENTINEL,
    build_answer_prompt,
    build_judge_prompt,
    parse_judge_verdict,
    run_qa,
)


class FakeRunner(LLMRunner):
    """Returns canned responses keyed by substring match against the prompt."""

    def __init__(self, responses: dict[str, str], default: str = ""):
        self.spec = "fake:test"
        self.responses = responses
        self.default = default
        self.prompts: list[str] = []

    def complete(self, prompt: str) -> LLMResult:
        self.prompts.append(prompt)
        for needle, response in self.responses.items():
            if needle in prompt:
                return LLMResult(
                    text=response, model="fake", input_tokens=10, output_tokens=5, latency_ms=1.0
                )
        if self.default:
            return LLMResult(
                text=self.default, model="fake", input_tokens=10, output_tokens=5, latency_ms=1.0
            )
        raise LLMRunnerError(f"No canned response for prompt: {prompt[:80]}")


def _row(
    query_id: str,
    question: str,
    expected: str | None,
    context: str,
    category: str = "single_hop",
) -> PerQueryRetrievalResult:
    return PerQueryRetrievalResult(
        provider="bm-local",
        query_id=query_id,
        query_text=question,
        category=category,
        ground_truth=[],
        expected_answer=expected,
        recall_at_5=0.0,
        recall_at_10=0.0,
        precision_at_5=0.0,
        mrr=0.0,
        content_hit=False,
        latency_ms=1.0,
        retrieved_context=context,
    )


class TestPrompts:
    def test_answer_prompt_includes_question_and_context(self):
        prompt = build_answer_prompt("Where does Joanna live?", "Joanna lives in Austin.")
        assert "Where does Joanna live?" in prompt
        assert "Joanna lives in Austin." in prompt
        assert ABSTAIN_SENTINEL in prompt

    def test_answer_prompt_marks_empty_context(self):
        prompt = build_answer_prompt("Where does Joanna live?", "   ")
        assert "(no memories were retrieved)" in prompt

    def test_judge_prompt_includes_all_parts(self):
        prompt = build_judge_prompt("Q?", "gold fact", "candidate fact")
        assert "Q?" in prompt
        assert "gold fact" in prompt
        assert "candidate fact" in prompt


class TestParseJudgeVerdict:
    def test_plain_json(self):
        correct, reason = parse_judge_verdict('{"correct": true, "reason": "matches"}')
        assert correct is True
        assert reason == "matches"

    def test_json_in_code_fence(self):
        raw = 'Here is my verdict:\n```json\n{"correct": false, "reason": "missing fact"}\n```'
        correct, reason = parse_judge_verdict(raw)
        assert correct is False
        assert reason == "missing fact"

    def test_no_json_raises(self):
        with pytest.raises(ValueError):
            parse_judge_verdict("The answer is correct.")

    def test_non_boolean_correct_raises(self):
        with pytest.raises(ValueError):
            parse_judge_verdict('{"correct": "yes"}')


class TestRunQA:
    def test_correct_and_incorrect_cases(self):
        rows = [
            _row("q1", "Where does Joanna live?", "Austin", "Joanna lives in Austin."),
            _row("q2", "What is Anthony's job?", "Engineer", "Anthony enjoys hiking."),
        ]
        answerer = FakeRunner(
            {
                "Where does Joanna live?": "Austin",
                "What is Anthony's job?": ABSTAIN_SENTINEL,
            }
        )
        judge = FakeRunner(
            {
                "Candidate answer: Austin": '{"correct": true, "reason": "match"}',
                f"Candidate answer: {ABSTAIN_SENTINEL}": '{"correct": false, "reason": "abstained on answerable"}',
            }
        )

        cases, summary = run_qa(
            rows, provider="bm-local", answerer=answerer, judge=judge, max_workers=1
        )

        assert summary.total_cases == 2
        assert summary.correct_count == 1
        assert summary.accuracy == 0.5
        assert summary.abstain_count == 1
        by_id = {case.query_id: case for case in cases}
        assert by_id["q1"].correct is True
        assert by_id["q2"].correct is False
        assert by_id["q2"].abstained is True

    def test_category_breakdown(self):
        rows = [
            _row("q1", "Q1?", "A1", "ctx", category="single_hop"),
            _row("q2", "Q2?", "A2", "ctx", category="temporal"),
        ]
        answerer = FakeRunner({}, default="some answer")
        judge = FakeRunner(
            {
                "Q1?": '{"correct": true, "reason": "ok"}',
                "Q2?": '{"correct": false, "reason": "wrong"}',
            }
        )
        _, summary = run_qa(
            rows, provider="bm-local", answerer=answerer, judge=judge, max_workers=1
        )

        assert summary.by_category["single_hop"].accuracy == 1.0
        assert summary.by_category["temporal"].accuracy == 0.0

    def test_rows_without_expected_answer_are_skipped(self):
        rows = [_row("q1", "Q1?", None, "ctx")]
        answerer = FakeRunner({})
        judge = FakeRunner({})
        cases, summary = run_qa(rows, provider="bm-local", answerer=answerer, judge=judge)
        assert cases == []
        assert summary.skipped_reason is not None

    def test_runner_error_recorded_not_raised(self):
        rows = [
            _row("q1", "Q1?", "A1", "ctx"),
            _row("q2", "Q2?", "A2", "ctx"),
        ]
        answerer = FakeRunner({"Q2?": "answer two"})  # Q1 has no canned response -> error
        judge = FakeRunner({}, default='{"correct": true, "reason": "ok"}')

        cases, summary = run_qa(
            rows, provider="bm-local", answerer=answerer, judge=judge, max_workers=1
        )

        by_id = {case.query_id: case for case in cases}
        assert by_id["q1"].error is not None
        assert by_id["q1"].correct is False
        assert by_id["q2"].correct is True
        assert summary.error_count == 1
        assert summary.total_cases == 2

    def test_token_and_latency_accounting(self):
        rows = [_row("q1", "Q1?", "A1", "ctx")]
        answerer = FakeRunner({}, default="answer")
        judge = FakeRunner({}, default='{"correct": true, "reason": "ok"}')
        _, summary = run_qa(rows, provider="bm-local", answerer=answerer, judge=judge)
        assert summary.total_answer_input_tokens == 10
        assert summary.total_answer_output_tokens == 5
        assert summary.mean_answer_latency_ms == pytest.approx(1.0)


class TestQAStageArtifacts:
    def test_run_qa_stage_writes_artifacts(self, tmp_path, monkeypatch):
        from basic_memory_benchmarks import runner as runner_module

        rows = [
            _row("q1", "Where does Joanna live?", "Austin", "Joanna lives in Austin."),
        ]
        retrieval_path = tmp_path / "per-query-retrieval.jsonl"
        with retrieval_path.open("w", encoding="utf-8") as file:
            for row in rows:
                file.write(json.dumps(row.model_dump(mode="json")) + "\n")

        fake = FakeRunner({}, default='{"correct": true, "reason": "ok"}')
        monkeypatch.setattr("basic_memory_benchmarks.llm.runners.create_runner", lambda spec: fake)

        runner_module.run_qa_stage(
            run_dir=tmp_path,
            answerer_spec="fake:test",
            judge_spec="fake:test",
            max_workers=1,
        )

        assert (tmp_path / "per-query-qa.jsonl").exists()
        summary = json.loads((tmp_path / "qa-summary.json").read_text())
        assert summary["providers"][0]["provider"] == "bm-local"
        assert summary["providers"][0]["total_cases"] == 1


class TestQuestionDate:
    def test_question_date_reaches_answerer_and_judge(self):
        row = _row("q1", "How many weeks ago did I visit the dentist?", "Three weeks ago", "ctx")
        row = row.model_copy(update={"metadata": {"question_date": "2023/05/30 (Tue) 23:40"}})
        answerer = FakeRunner({}, default="Three weeks ago")
        judge = FakeRunner({}, default='{"correct": true, "reason": "ok"}')

        run_qa([row], provider="bm-local", answerer=answerer, judge=judge, max_workers=1)

        assert "question asked on 2023/05/30 (Tue) 23:40" in answerer.prompts[0]
        assert "question asked on 2023/05/30 (Tue) 23:40" in judge.prompts[0]

    def test_no_date_means_plain_question(self):
        row = _row("q1", "Where does Joanna live?", "Austin", "ctx")
        answerer = FakeRunner({}, default="Austin")
        judge = FakeRunner({}, default='{"correct": true, "reason": "ok"}')

        run_qa([row], provider="bm-local", answerer=answerer, judge=judge, max_workers=1)

        assert "question asked on" not in answerer.prompts[0]
