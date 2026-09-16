import pandas as pd
import pytest

from app.query_engine import QueryEngine


@pytest.fixture
def query_engine():
    return QueryEngine()


def test_count_tickets(query_engine):
    assert query_engine.count_tickets() == 500


def test_filter_tickets_by_status(query_engine):
    open_tickets = query_engine.filter_tickets(status="Open")
    assert len(open_tickets) == query_engine.count_tickets(status="Open")
    assert open_tickets["status"].eq("Open").all()


def test_filter_tickets_rejects_unknown_filter(query_engine):
    with pytest.raises(ValueError, match="Unsupported filter 'ticket_type'"):
        query_engine.filter_tickets(ticket_type="Critical")


def test_average_metrics_exclude_missing_values(query_engine):
    assert query_engine.average_customer_rating() > 0
    assert query_engine.average_resolution_time() > 0
    assert pd.notna(query_engine.average_customer_rating())
    assert pd.notna(query_engine.average_resolution_time())


def test_group_counts_by_category_and_priority(query_engine):
    category_counts = query_engine.group_by_category()
    priority_counts = query_engine.group_by_priority()

    assert category_counts.sum() == 500
    assert priority_counts.sum() == 500
    assert set(category_counts.index) == {"Billing", "General", "Technical"}
    assert set(priority_counts.index) == {"Low", "Medium", "High", "Critical"}


def test_find_unresolved_and_threshold_cases(query_engine):
    unresolved = query_engine.find_unresolved_tickets()
    critical_unresolved = query_engine.find_critical_or_high_unresolved()
    slow_tickets = query_engine.find_tickets_with_resolution_time_above(24)

    assert unresolved["status"].eq("Open").any()
    assert critical_unresolved["priority"].isin(["Critical", "High"]).all()
    assert (slow_tickets["resolution_time_hrs"] > 24).all()


def test_agent_with_most_resolved_tickets(query_engine):
    top_agent = query_engine.agent_with_most_resolved_tickets()

    assert isinstance(top_agent, str)
    assert len(top_agent) > 0
    assert top_agent in query_engine.filter_tickets(status="Resolved")["agent_id"].unique()
