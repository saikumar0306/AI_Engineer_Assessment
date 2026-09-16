from __future__ import annotations

import os
from datetime import datetime
from typing import Any

import pandas as pd
import requests
import streamlit as st


API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000").rstrip("/")
REQUEST_TIMEOUT_SECONDS = 30

EXAMPLE_QUESTIONS = [
    "How many tickets are open?",
    "How many Critical tickets are unresolved?",
    "What is the average resolution time for Technical tickets?",
    "Which category has the most tickets?",
    "Which agent resolved the most tickets?",
]

OVERVIEW_QUESTIONS = {
    "Total tickets": "How many tickets are there?",
    "Open tickets": "How many tickets are open?",
    "Resolved tickets": "How many tickets are resolved?",
    "Escalated tickets": "How many tickets are escalated?",
    "Critical tickets": "How many Critical tickets are there?",
}


def _api_url(path: str) -> str:
    return f"{API_BASE_URL}/{path.lstrip('/')}"


def _error_message(error: requests.RequestException) -> str:
    response = getattr(error, "response", None)
    if response is not None:
        try:
            detail = response.json().get("detail")
        except ValueError:
            detail = None
        if detail:
            if response.status_code == 503:
                return f"Ollama is unavailable: {detail}"
            return f"The API could not process that request: {detail}"
        return f"The API returned an error (HTTP {response.status_code})."
    return "The FastAPI service is not reachable. Start it and try again."


def _get(path: str, **params: Any) -> dict[str, Any]:
    response = requests.get(
        _api_url(path), params=params or None, timeout=REQUEST_TIMEOUT_SECONDS
    )
    response.raise_for_status()
    return response.json()


def _post(path: str, payload: dict[str, Any]) -> dict[str, Any]:
    response = requests.post(
        _api_url(path), json=payload, timeout=REQUEST_TIMEOUT_SECONDS
    )
    response.raise_for_status()
    return response.json()


@st.cache_data(ttl=30, show_spinner=False)
def load_overview() -> tuple[dict[str, Any], str | None]:
    metrics: dict[str, Any] = {}
    for label, question in OVERVIEW_QUESTIONS.items():
        try:
            metrics[label] = _post("/query", {"question": question}).get("result")
        except requests.RequestException as error:
            return {}, _error_message(error)
    return metrics, None


def display_result(result: Any) -> None:
    if isinstance(result, (dict, list)):
        st.json(result)
    else:
        st.metric("Result", str(result))


def render_overview() -> None:
    st.subheader("Dataset Overview")
    metrics, error = load_overview()
    if error:
        st.warning(f"Overview metrics are unavailable. {error}")
        return

    columns = st.columns(len(OVERVIEW_QUESTIONS))
    for column, label in zip(columns, OVERVIEW_QUESTIONS):
        column.metric(label, metrics.get(label, "Unavailable"))


def render_query() -> None:
    st.subheader("Ask the AI")
    st.caption("Ask a natural-language question about the support ticket dataset.")
    question = st.text_area(
        "Question",
        placeholder="Ask a question about the support tickets...",
        label_visibility="collapsed",
    )
    st.write("Example questions")
    selected_example = st.selectbox(
        "Example questions",
        ["Choose an example..."] + EXAMPLE_QUESTIONS,
        label_visibility="collapsed",
    )
    if selected_example != "Choose an example..." and not question:
        question = selected_example

    if st.button("Ask", type="primary"):
        if not question.strip():
            st.warning("Enter a question before asking the AI.")
            return
        try:
            response = _post("/query", {"question": question.strip()})
        except requests.RequestException as error:
            st.error(_error_message(error))
            return

        st.subheader("Answer")
        st.write(response.get("answer", "No answer was returned."))
        if response.get("result") is not None:
            st.write("Result")
            display_result(response["result"])


def render_anomalies() -> None:
    st.subheader("Detected Anomalies")
    use_reference_time = st.checkbox("Use a reference date/time")
    params: dict[str, str] = {}
    if use_reference_time:
        reference_time = st.datetime_input(
            "Reference date/time", value=datetime.now().replace(second=0, microsecond=0)
        )
        params["reference_time"] = reference_time.isoformat()

    if st.button("Refresh anomalies") or "anomaly_response" not in st.session_state:
        try:
            st.session_state.anomaly_response = _get("/anomalies", **params)
        except requests.RequestException as error:
            st.error(_error_message(error))
            return

    response = st.session_state.anomaly_response
    records = response.get("anomalies", [])
    long_resolution = sum(
        record.get("anomaly_type") == "long_resolution_time" for record in records
    )
    high_priority = sum(
        record.get("anomaly_type") == "high_priority_unresolved" for record in records
    )
    total, long_column, high_column = st.columns(3)
    total.metric("Total anomalies", response.get("total", len(records)))
    long_column.metric("Long resolution", long_resolution)
    high_column.metric("High-priority unresolved", high_priority)
    if records:
        st.dataframe(pd.DataFrame(records), width="stretch", hide_index=True)
    else:
        st.info("No anomalies were detected.")


def main() -> None:
    st.set_page_config(page_title="AI Support Ticket Intelligence", page_icon="🎫", layout="wide")
    st.title("AI Support Ticket Intelligence")
    st.write(
        "Analyze support tickets with natural-language questions and identify unusual "
        "resolution and priority patterns."
    )
    render_overview()
    st.divider()
    render_query()
    st.divider()
    render_anomalies()


if __name__ == "__main__":
    main()