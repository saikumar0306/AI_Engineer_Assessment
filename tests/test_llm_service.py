import json

import pytest

from app.llm_service import (
    OllamaService,
    StructuredTicketQuery,
    TicketQueryInterpreter,
    UnsupportedIntentError,
)
from app.query_engine import QueryEngine


class MockResponse:
    def __init__(self, payload):
        self.status_code = 200
        self._payload = payload

    def json(self):
        return self._payload


class MockErrorResponse:
    def __init__(self, status_code=500, payload=None):
        self.status_code = status_code
        self._payload = payload or {"error": "bad model"}

    def json(self):
        return self._payload


def test_valid_llm_structured_output(monkeypatch):
    payload = {
        "intent": "count",
        "filters": {"priority": "Critical", "status": ["Open", "Escalated"]},
    }
    monkeypatch.setattr(
        "app.llm_service.requests.post",
        lambda *args, **kwargs: MockResponse(payload),
    )

    service = OllamaService(model="llama3.2:3b")
    parsed = service.interpret_question("How many critical tickets are still unresolved?")

    assert isinstance(parsed, StructuredTicketQuery)
    assert parsed.intent == "count"
    assert parsed.filters["priority"] == "Critical"
    assert parsed.filters["status"] == ["Open", "Escalated"]


def test_ollama_request_uses_closed_structured_schema(monkeypatch):
    captured = {}

    def mock_post(*args, **kwargs):
        captured.update(kwargs["json"])
        return MockResponse({"intent": "count", "filters": {"status": ["Open"]}})

    monkeypatch.setattr("app.llm_service.requests.post", mock_post)

    OllamaService(model="llama3.2:3b").interpret_question("How many tickets are open?")

    assert captured["format"]["additionalProperties"] is False
    assert captured["format"]["properties"]["filters"]["additionalProperties"] is False
    assert "threshold" in captured["prompt"]
    assert "Unresolved always means status ['Open', 'Escalated']" in captured["prompt"]


def test_invalid_llm_output(monkeypatch):
    monkeypatch.setattr(
        "app.llm_service.requests.post",
        lambda *args, **kwargs: MockResponse({"intent": "count"}),
    )

    service = OllamaService(model="llama3.2:3b")
    with pytest.raises(ValueError, match="missing|required|invalid"):
        service.interpret_question("How many critical tickets are still unresolved?")


def test_unsupported_llm_filter_is_rejected(monkeypatch):
    payload = {"intent": "count", "filters": {"ticket_type": "Critical"}}
    monkeypatch.setattr(
        "app.llm_service.requests.post",
        lambda *args, **kwargs: MockResponse(payload),
    )

    service = OllamaService(model="llama3.2:3b")
    with pytest.raises(ValueError, match="Unsupported filter"):
        service.interpret_question("How many Critical tickets are there?")


def test_non_object_llm_filters_are_rejected(monkeypatch):
    payload = {"intent": "count", "filters": ["status", "Open"]}
    monkeypatch.setattr(
        "app.llm_service.requests.post",
        lambda *args, **kwargs: MockResponse(payload),
    )

    service = OllamaService(model="llama3.2:3b")
    with pytest.raises(ValueError, match="filters must be an object"):
        service.interpret_question("How many tickets are open?")


def test_unsupported_intent(monkeypatch):
    payload = {"intent": "totally_unknown", "filters": {}}
    monkeypatch.setattr(
        "app.llm_service.requests.post",
        lambda *args, **kwargs: MockResponse(payload),
    )

    service = OllamaService(model="llama3.2:3b")
    with pytest.raises(UnsupportedIntentError):
        service.interpret_question("Please tell me something random")


@pytest.mark.parametrize(
    "question, expected_intent, expected_filters",
    [
        (
            "How many tickets are open?",
            "count",
            {"status": ["Open"]},
        ),
        (
            "How many critical tickets are still unresolved?",
            "count",
            {"priority": "Critical", "status": ["Open", "Escalated"]},
        ),
        (
            "What is the average resolution time for technical tickets?",
            "average_resolution_time",
            {"category": "Technical"},
        ),
        (
            "Which category has the most tickets?",
            "group_by_category",
            {},
        ),
        (
            "Who is the top resolved agent?",
            "top_resolved_agent",
            {},
        ),
        (
            "How many Escalated tickets are there?",
            "count",
            {"status": ["Escalated"]},
        ),
    ],
)
def test_correct_mapping_of_natural_language_questions(monkeypatch, question, expected_intent, expected_filters):
    payload = {"intent": expected_intent, "filters": expected_filters}
    monkeypatch.setattr(
        "app.llm_service.requests.post",
        lambda *args, **kwargs: MockResponse(payload),
    )

    service = OllamaService(model="llama3.2:3b")
    parsed = service.interpret_question(question)

    assert parsed.intent == expected_intent
    assert parsed.filters == expected_filters


def test_invalid_filter_values_are_rejected():
    with pytest.raises(ValueError, match="Invalid category"):
        StructuredTicketQuery(intent="count", filters={"category": "Critical"})


def test_threshold_is_restricted_to_threshold_intent():
    with pytest.raises(ValueError, match="threshold filter"):
        StructuredTicketQuery(intent="count", filters={"threshold": 24})


def test_end_to_end_query_to_deterministic_result(monkeypatch):
    engine = QueryEngine()
    payload = {
        "intent": "count",
        "filters": {"priority": "Critical", "status": ["Open", "Escalated"]},
    }
    monkeypatch.setattr(
        "app.llm_service.requests.post",
        lambda *args, **kwargs: MockResponse(payload),
    )

    interpreter = TicketQueryInterpreter(query_engine=engine)
    result = interpreter.answer_question("How many critical tickets are still unresolved?")

    expected = engine.count_tickets(priority="Critical", status=["Open", "Escalated"])
    assert str(expected) in result
    assert "critical" in result.lower()
