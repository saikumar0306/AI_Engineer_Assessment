import pandas as pd
import pytest

from app.data_loader import EXPECTED_COLUMNS, load_tickets, validate_tickets


def make_tickets() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ticket_id": ["TKT-001", "TKT-002"],
            "created_at": pd.to_datetime(["2024-01-01 09:00", "2024-01-02 10:30"]),
            "category": ["Billing", "Technical"],
            "priority": ["High", "Critical"],
            "status": ["Resolved", "Open"],
            "response_time_hrs": [1.5, 2.0],
            "resolution_time_hrs": [4.0, float("nan")],
            "agent_id": ["AGT-01", "AGT-02"],
            "customer_rating": [5.0, float("nan")],
            "issue_summary": ["Incorrect charge", "Login failure"],
        }
    )


def test_load_actual_dataset_has_expected_schema_and_types():
    tickets = load_tickets()

    assert list(tickets.columns) == EXPECTED_COLUMNS
    assert len(tickets) == 500
    assert pd.api.types.is_datetime64_ns_dtype(tickets["created_at"])
    assert pd.api.types.is_float_dtype(tickets["response_time_hrs"])
    assert pd.api.types.is_float_dtype(tickets["resolution_time_hrs"])
    assert pd.api.types.is_float_dtype(tickets["customer_rating"])


def test_load_actual_dataset_preserves_missing_values():
    tickets = load_tickets()

    assert tickets["resolution_time_hrs"].isna().sum() == 173
    assert tickets["customer_rating"].isna().sum() == 173


def test_validate_tickets_accepts_nullable_resolution_and_rating():
    validate_tickets(make_tickets())


def test_validate_tickets_rejects_duplicate_ticket_ids():
    tickets = make_tickets()
    tickets.loc[1, "ticket_id"] = tickets.loc[0, "ticket_id"]

    with pytest.raises(ValueError, match="unique"):
        validate_tickets(tickets)


def test_validate_tickets_rejects_invalid_numeric_values():
    tickets = make_tickets()
    tickets.loc[0, "response_time_hrs"] = -1

    with pytest.raises(ValueError, match="negative"):
        validate_tickets(tickets)


def test_validate_tickets_rejects_invalid_categories():
    tickets = make_tickets()
    tickets.loc[0, "category"] = "Unknown"

    with pytest.raises(ValueError, match="Invalid values"):
        validate_tickets(tickets)