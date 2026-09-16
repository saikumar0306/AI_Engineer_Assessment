from datetime import datetime
from types import SimpleNamespace

from fastapi.testclient import TestClient

from app import main
from app.llm_service import StructuredTicketQuery, TicketQueryInterpreter


client = TestClient(main.app)


class FakeInterpreter:
    def __init__(self, result=None, error=None):
        self.result = result
        self.error = error
        self.llm_service = self

    def interpret_question(self, question):
        if self.error:
            raise self.error
        return StructuredTicketQuery(intent="count", filters={})

    def execute(self, query):
        return self.result

    def format_answer(self, query, result):
        return f"There are {result} ticket(s)."


class FakeDetector:
    def __init__(self, records):
        self.records = records
        self.reference_time = None

    def find_all_anomalies(self, reference_time=None):
        self.reference_time = reference_time
        return self.records


class UnsupportedFilterLLM:
    def interpret_question(self, question):
        return SimpleNamespace(intent="count", filters={"ticket_type": "Critical"})


def test_health():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_query_uses_mocked_llm_and_returns_result(monkeypatch):
    monkeypatch.setattr(main, "query_interpreter", FakeInterpreter(result=3))

    response = client.post("/query", json={"question": "How many tickets?"})

    assert response.status_code == 200
    assert response.json() == {
        "question": "How many tickets?",
        "answer": "There are 3 ticket(s).",
        "result": 3,
    }


def test_query_rejects_empty_question():
    response = client.post("/query", json={"question": "   "})

    assert response.status_code == 422
    assert "Question must not be empty" in response.json()["detail"][0]["msg"]


def test_query_maps_ollama_error(monkeypatch):
    monkeypatch.setattr(main, "query_interpreter", FakeInterpreter(error=RuntimeError("Ollama unavailable")))

    response = client.post("/query", json={"question": "How many tickets?"})

    assert response.status_code == 503
    assert response.json()["detail"] == "Ollama unavailable"


def test_query_maps_invalid_llm_response(monkeypatch):
    monkeypatch.setattr(main, "query_interpreter", FakeInterpreter(error=ValueError("Invalid LLM JSON output.")))

    response = client.post("/query", json={"question": "How many tickets?"})

    assert response.status_code == 502
    assert response.json()["detail"] == "Invalid LLM JSON output."


def test_anomalies_returns_records_and_accepts_reference_time(monkeypatch):
    detector = FakeDetector(
        [
            {
                "ticket_id": "T-1",
                "anomaly_type": "high_priority_unresolved",
                "reason": "older than 24 hours",
                "priority": "Critical",
                "status": "Open",
                "created_at": datetime(2024, 1, 1),
            }
        ]
    )
    monkeypatch.setattr(main, "anomaly_detector", detector)

    response = client.get("/anomalies", params={"reference_time": "2024-01-03T00:00:00"})

    assert response.status_code == 200
    assert response.json()["total"] == 1
    assert response.json()["anomalies"][0]["ticket_id"] == "T-1"
    assert detector.reference_time == datetime(2024, 1, 3)


def test_query_maps_internal_errors(monkeypatch):
    monkeypatch.setattr(main, "query_interpreter", FakeInterpreter(result=object()))

    response = client.post("/query", json={"question": "How many tickets?"})

    assert response.status_code == 500
    assert response.json()["detail"] == "Internal query error."


def test_query_maps_unsupported_llm_filter(monkeypatch):
    interpreter = TicketQueryInterpreter(llm_service=UnsupportedFilterLLM())
    monkeypatch.setattr(main, "query_interpreter", interpreter)

    response = client.post("/query", json={"question": "How many Critical tickets are there?"})

    assert response.status_code == 502
    assert response.json()["detail"] == "Unsupported filter 'ticket_type'."
