from __future__ import annotations

import json
import os
import re
from typing import Any

import requests
from pydantic import BaseModel, Field, ValidationError, field_validator, model_validator

from app.query_engine import QueryEngine

SUPPORTED_INTENTS = {
    "count",
    "average_response_time",
    "average_resolution_time",
    "average_customer_rating",
    "group_by_category",
    "group_by_priority",
    "group_by_status",
    "group_by_agent",
    "unresolved_tickets",
    "critical_high_unresolved",
    "resolution_time_above",
    "top_resolved_agent",
}
SUPPORTED_FILTERS = {
    "ticket_id",
    "created_at",
    "category",
    "priority",
    "status",
    "response_time_hrs",
    "resolution_time_hrs",
    "agent_id",
    "customer_rating",
    "issue_summary",
    "threshold",
}
ALLOWED_CATEGORIES = {"Billing", "General", "Technical"}
ALLOWED_PRIORITIES = {"Low", "Medium", "High", "Critical"}
ALLOWED_STATUSES = {"Open", "Escalated", "Resolved"}

OLLAMA_RESPONSE_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["intent", "filters"],
    "properties": {
        "intent": {"type": "string", "enum": sorted(SUPPORTED_INTENTS)},
        "filters": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "ticket_id": {"type": "string"},
                "created_at": {"type": "string"},
                "category": {"type": "string", "enum": sorted(ALLOWED_CATEGORIES)},
                "priority": {"type": "string", "enum": sorted(ALLOWED_PRIORITIES)},
                "status": {
                    "type": "array",
                    "items": {"type": "string", "enum": sorted(ALLOWED_STATUSES)},
                },
                "response_time_hrs": {"type": "number"},
                "resolution_time_hrs": {"type": "number"},
                "agent_id": {"type": "string"},
                "customer_rating": {"type": "number"},
                "issue_summary": {"type": "string"},
                "threshold": {"type": "number"},
            },
        },
    },
}


class UnsupportedIntentError(ValueError):
    """Raised when the LLM produces an intent outside the supported ticket analysis set."""


class StructuredTicketQuery(BaseModel):
    intent: str
    filters: dict[str, Any] = Field(default_factory=dict)

    @field_validator("intent")
    @classmethod
    def validate_intent(cls, value: str) -> str:
        normalized = value.strip()
        if normalized not in SUPPORTED_INTENTS:
            raise ValueError(f"Unsupported intent '{value}'.")
        return normalized

    @model_validator(mode="after")
    def validate_filters(self) -> "StructuredTicketQuery":
        if self.filters is None:
            raise ValueError("Missing required field: filters")
        if not isinstance(self.filters, dict):
            raise ValueError("Invalid LLM output: filters must be an object.")
        unsupported = sorted(set(self.filters) - SUPPORTED_FILTERS)
        if unsupported:
            raise ValueError(f"Unsupported filter(s): {', '.join(unsupported)}.")

        category = self.filters.get("category")
        if category is not None and category not in ALLOWED_CATEGORIES:
            raise ValueError(f"Invalid category '{category}'.")

        priority = self.filters.get("priority")
        if priority is not None and priority not in ALLOWED_PRIORITIES:
            raise ValueError(f"Invalid priority '{priority}'.")

        statuses = self.filters.get("status")
        if statuses is not None:
            if not isinstance(statuses, list) or any(status not in ALLOWED_STATUSES for status in statuses):
                raise ValueError("Invalid status filter.")

        if "threshold" in self.filters and self.intent != "resolution_time_above":
            raise ValueError("The threshold filter is only valid for resolution_time_above.")
        return self


class OllamaService:
    """Interprets natural-language ticket questions into supported structured queries."""

    def __init__(self, base_url: str | None = None, model: str | None = None, timeout: int = 30):
        self.base_url = (base_url or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")).rstrip("/")
        self.model = model or os.getenv("OLLAMA_MODEL", "llama3.2:3b")
        self.timeout = timeout

    def _call_ollama(self, question: str) -> dict[str, Any]:
        url = f"{self.base_url}/api/generate"
        payload = {
            "model": self.model,
            "prompt": self._build_prompt(question),
            "stream": False,
            "format": OLLAMA_RESPONSE_SCHEMA,
        }

        try:
            response = requests.post(url, json=payload, timeout=self.timeout)
        except requests.RequestException as exc:  # pragma: no cover - network layer path
            raise RuntimeError(f"Could not reach Ollama at {url}: {exc}") from exc

        if response.status_code != 200:
            body = response.text
            try:
                error_payload = response.json()
                error_message = error_payload.get("error")
            except ValueError:
                error_message = None
            raise RuntimeError(
                f"Ollama request failed with status {response.status_code}: "
                f"{error_message or body or 'unknown error'}"
            )

        try:
            result = response.json()
        except ValueError as exc:
            raise ValueError("Invalid LLM JSON response from Ollama.") from exc

        if not isinstance(result, dict):
            raise ValueError("Invalid LLM JSON response from Ollama.")

        if "error" in result:
            raise RuntimeError(f"Ollama reported an error: {result['error']}")

        return result

    def _build_prompt(self, question: str) -> str:
        return (
            "You are a strict support-ticket intent classifier. "
            "Interpret the user's question and return ONLY one JSON object with exactly these keys: "
            "'intent' and 'filters'. "
            f"The 'intent' must be one of: {', '.join(sorted(SUPPORTED_INTENTS))}. "
            "Allowed filter names are exactly: ticket_id, created_at, category, priority, status, "
            "response_time_hrs, resolution_time_hrs, agent_id, customer_rating, issue_summary, threshold. "
            "Never invent filter names, never use 'all', and omit filters that are not needed. "
            "category must be exactly Billing, General, or Technical. "
            "priority must be exactly Low, Medium, High, or Critical. "
            "status must be an array containing only Open, Escalated, or Resolved. "
            "Unresolved always means status ['Open', 'Escalated']; it never means category or priority. "
            "Use threshold only with resolution_time_above. Do not calculate the answer. "
            "Use these mappings exactly: "
            "open ticket count -> {'intent':'count','filters':{'status':['Open']}}; "
            "Critical unresolved count -> {'intent':'count','filters':{'priority':'Critical','status':['Open','Escalated']}}; "
            "Technical average resolution time -> {'intent':'average_resolution_time','filters':{'category':'Technical'}}; "
            "most tickets by category -> {'intent':'group_by_category','filters':{}}; "
            "agent who resolved the most -> {'intent':'top_resolved_agent','filters':{}}; "
            "Escalated ticket count -> {'intent':'count','filters':{'status':['Escalated']}}. "
            "Return JSON only. "
            f"User question: {question}"
        )

    def _extract_json_payload(self, raw_response: dict[str, Any]) -> dict[str, Any]:
        if "response" in raw_response:
            candidate = raw_response["response"]
        elif "content" in raw_response:
            candidate = raw_response["content"]
        else:
            candidate = json.dumps(raw_response)

        if not isinstance(candidate, str):
            candidate = json.dumps(candidate)

        candidate = candidate.strip()
        if not candidate:
            raise ValueError("Empty LLM response.")

        if candidate.startswith("```"):
            candidate = re.sub(r"^```(?:json)?\s*|```\s*$", "", candidate, flags=re.IGNORECASE).strip()
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError as exc:
            raise ValueError("Invalid LLM JSON output.") from exc

        if not isinstance(parsed, dict):
            raise ValueError("Invalid LLM JSON output: expected an object.")

        return parsed

    def _normalize_filters(self, filters: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(filters, dict):
            raise ValueError("Invalid LLM output: filters must be an object.")

        normalized: dict[str, Any] = {}

        for key, value in filters.items():
            normalized_key = str(key).strip()
            if normalized_key == "status":
                if isinstance(value, str):
                    value = [value]
                if isinstance(value, list):
                    statuses = []
                    for item in value:
                        status_value = str(item).strip()
                        if status_value.lower() == "unresolved":
                            statuses.extend(["Open", "Escalated"])
                        else:
                            statuses.append(status_value)
                    normalized[normalized_key] = list(dict.fromkeys(statuses))
                    continue
            elif normalized_key == "threshold" and value is not None:
                normalized[normalized_key] = float(value)
                continue

            normalized[normalized_key] = value

        return normalized

    def interpret_question(self, question: str) -> StructuredTicketQuery:
        raw_response = self._call_ollama(question)
        payload = self._extract_json_payload(raw_response)

        if "intent" not in payload:
            raise ValueError("Invalid LLM output: missing required field 'intent'.")
        if "filters" not in payload or payload.get("filters") is None:
            raise ValueError("Invalid LLM output: missing required field 'filters'.")

        intent = payload.get("intent")
        filters = payload.get("filters", {})

        try:
            parsed = StructuredTicketQuery(intent=str(intent), filters=self._normalize_filters(filters))
        except ValidationError as exc:
            message = exc.errors()[0]["msg"] if exc.errors() else str(exc)
            if "Unsupported intent" in message:
                raise UnsupportedIntentError("I couldn't map that question to a supported ticket analysis.") from exc
            raise ValueError(f"Invalid LLM output: {message}") from exc

        if parsed.intent not in SUPPORTED_INTENTS:
            raise UnsupportedIntentError("I couldn't map that question to a supported ticket analysis.")

        return parsed


class TicketQueryInterpreter:
    """Maps interpreted intent to the deterministic QueryEngine behavior."""

    def __init__(self, query_engine: QueryEngine | None = None, llm_service: OllamaService | None = None):
        self.query_engine = query_engine or QueryEngine()
        self.llm_service = llm_service or OllamaService()

    def execute(self, query: StructuredTicketQuery) -> Any:
        filters = dict(query.filters)

        if query.intent == "count":
            return self.query_engine.count_tickets(**filters)
        if query.intent == "average_response_time":
            return self.query_engine.average_response_time(**filters)
        if query.intent == "average_resolution_time":
            return self.query_engine.average_resolution_time(**filters)
        if query.intent == "average_customer_rating":
            return self.query_engine.average_customer_rating(**filters)
        if query.intent == "group_by_category":
            data = self.query_engine.filter_tickets(**filters)
            return data["category"].value_counts().sort_values(ascending=False)
        if query.intent == "group_by_priority":
            data = self.query_engine.filter_tickets(**filters)
            return data["priority"].value_counts().sort_values(ascending=False)
        if query.intent == "group_by_status":
            data = self.query_engine.filter_tickets(**filters)
            return data["status"].value_counts().sort_values(ascending=False)
        if query.intent == "group_by_agent":
            data = self.query_engine.filter_tickets(**filters)
            return data["agent_id"].value_counts().sort_values(ascending=False)
        if query.intent == "unresolved_tickets":
            if "status" in filters and isinstance(filters["status"], list):
                return self.query_engine.filter_tickets(**filters)
            return self.query_engine.find_unresolved_tickets()
        if query.intent == "critical_high_unresolved":
            return self.query_engine.find_critical_or_high_unresolved()
        if query.intent == "resolution_time_above":
            threshold = filters.get("threshold")
            if threshold is None:
                raise ValueError("Missing required threshold filter for resolution_time_above.")
            return self.query_engine.find_tickets_with_resolution_time_above(float(threshold))
        if query.intent == "top_resolved_agent":
            return self.query_engine.agent_with_most_resolved_tickets()

        raise UnsupportedIntentError(f"I couldn't map that question to a supported ticket analysis.")

    def _count_phrase(self, query: StructuredTicketQuery) -> str:
        filters = query.filters
        priority = filters.get("priority")
        status = filters.get("status")
        qualifiers: list[str] = []

        if priority:
            qualifiers.append(str(priority).lower())

        if isinstance(status, list):
            normalized_status = [str(item).strip() for item in status]
            if set(normalized_status) == {"Open", "Escalated"}:
                qualifiers.append("unresolved")
            elif normalized_status:
                qualifiers.append(normalized_status[0].lower())

        if not qualifiers:
            return ""

        return " ".join(qualifiers) + " "

    def answer_question(self, question: str) -> str:
        try:
            query = self.llm_service.interpret_question(question)
            result = self.execute(query)
        except (ValueError, RuntimeError, UnsupportedIntentError) as exc:
            return str(exc)

        return self.format_answer(query, result)

    def format_answer(self, query: StructuredTicketQuery, result: Any) -> str:
        """Format a deterministic query result for API and conversational callers."""

        if query.intent == "count":
            label = self._count_phrase(query)
            return f"There are {result} {label}ticket(s)."
        if query.intent in {"average_response_time", "average_resolution_time", "average_customer_rating"}:
            return f"The average is {float(result):.2f}."
        if query.intent in {"group_by_category", "group_by_priority", "group_by_status", "group_by_agent"}:
            return str(result.to_dict())
        if query.intent in {"unresolved_tickets", "critical_high_unresolved", "resolution_time_above"}:
            return f"Found {len(result)} matching tickets."
        if query.intent == "top_resolved_agent":
            return f"The top resolved agent is {result}."

        return str(result)


__all__ = [
    "OllamaService",
    "StructuredTicketQuery",
    "TicketQueryInterpreter",
    "UnsupportedIntentError",
]
