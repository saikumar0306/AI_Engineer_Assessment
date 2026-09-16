from datetime import datetime, timedelta

import pandas as pd
import pytest

from app.anomaly_detector import AnomalyDetector


@pytest.fixture
def sample_tickets_df():
    return pd.DataFrame(
        {
            "ticket_id": ["T-001", "T-002", "T-003", "T-004", "T-005", "T-006"],
            "created_at": pd.to_datetime(
                [
                    "2024-01-01 00:00",
                    "2024-01-02 12:00",
                    "2024-01-03 00:00",
                    "2024-01-04 04:00",
                    "2024-01-05 00:00",
                    "2024-01-06 00:00",
                ]
            ),
            "category": ["Technical", "Billing", "Technical", "General", "Billing", "Technical"],
            "priority": ["Low", "High", "Critical", "High", "Low", "Critical"],
            "status": ["Resolved", "Open", "Escalated", "Open", "Resolved", "Escalated"],
            "response_time_hrs": [1.0, 2.0, 3.0, 1.5, 2.5, 4.0],
            "resolution_time_hrs": [10.0, 12.0, 13.0, 14.0, 15.0, 30.0],
            "agent_id": ["A1", "A2", "A3", "A2", "A1", "A3"],
            "customer_rating": [4.5, 4.0, 3.8, 4.1, 4.3, 2.5],
            "issue_summary": ["a", "b", "c", "d", "e", "f"],
        }
    )


def test_iqr_calculation(sample_tickets_df):
    detector = AnomalyDetector()
    detector.tickets = sample_tickets_df

    q1, q3, iqr, upper_fence = detector.calculate_resolution_iqr()

    assert q1 == pytest.approx(12.25)
    assert q3 == pytest.approx(14.75)
    assert iqr == pytest.approx(2.5)
    assert upper_fence == pytest.approx(18.5)


def test_long_resolution_anomaly_detection(sample_tickets_df):
    detector = AnomalyDetector()
    detector.tickets = sample_tickets_df

    anomalies = detector.find_long_resolution_time_anomalies()

    assert len(anomalies) == 1
    assert anomalies[0]["ticket_id"] == "T-006"
    assert anomalies[0]["anomaly_type"] == "long_resolution_time"
    assert anomalies[0]["resolution_time_hrs"] == pytest.approx(30.0)


def test_high_priority_unresolved_detection(sample_tickets_df):
    detector = AnomalyDetector()
    detector.tickets = sample_tickets_df

    anomalies = detector.find_high_priority_unresolved_anomalies(
        reference_time=pd.Timestamp("2024-01-08 00:00")
    )

    assert len(anomalies) == 4
    assert {a["ticket_id"] for a in anomalies} == {"T-002", "T-003", "T-004", "T-006"}
    assert all(a["anomaly_type"] == "high_priority_unresolved" for a in anomalies)


def test_missing_resolution_times_are_ignored(sample_tickets_df):
    df = sample_tickets_df.copy()
    df.loc[0, "resolution_time_hrs"] = None
    detector = AnomalyDetector()
    detector.tickets = df

    anomalies = detector.find_long_resolution_time_anomalies()
    assert "T-006" in {a["ticket_id"] for a in anomalies}
    assert "T-001" not in {a["ticket_id"] for a in anomalies}


def test_configurable_reference_time(sample_tickets_df):
    detector = AnomalyDetector()
    detector.tickets = sample_tickets_df

    anomalies = detector.find_high_priority_unresolved_anomalies(
        reference_time=pd.Timestamp("2024-01-06 12:00")
    )

    assert {a["ticket_id"] for a in anomalies} == {"T-002", "T-003", "T-004"}


def test_correct_anomaly_output_structure(sample_tickets_df):
    detector = AnomalyDetector()
    detector.tickets = sample_tickets_df

    anomaly = detector.find_long_resolution_time_anomalies()[0]

    assert set(anomaly.keys()) == {"ticket_id", "anomaly_type", "reason", "resolution_time_hrs"}
    assert "IQR" in anomaly["reason"]
    assert anomaly["ticket_id"] == "T-006"


def test_no_duplicate_anomalies():
    detector = AnomalyDetector()
    detector.tickets = pd.DataFrame(
        {
            "ticket_id": ["T-100", "T-100", "T-101"],
            "created_at": pd.to_datetime(
                [
                    "2024-01-01 00:00",
                    "2024-01-01 00:00",
                    "2024-01-03 00:00",
                ]
            ),
            "category": ["Technical", "Technical", "Billing"],
            "priority": ["Critical", "Critical", "High"],
            "status": ["Open", "Open", "Escalated"],
            "response_time_hrs": [1.0, 1.0, 2.0],
            "resolution_time_hrs": [50.0, 50.0, 5.0],
            "agent_id": ["A1", "A1", "A2"],
            "customer_rating": [3.0, 3.0, 4.0],
            "issue_summary": ["dup", "dup", "other"],
        }
    )

    anomalies = detector.find_all_anomalies(reference_time=pd.Timestamp("2024-01-10 00:00"))
    ticket_counts = {
        a["ticket_id"]: 0 for a in anomalies if a["anomaly_type"] == "long_resolution_time"
    }
    for anomaly in anomalies:
        if anomaly["anomaly_type"] == "long_resolution_time":
            ticket_counts[anomaly["ticket_id"]] += 1

    assert all(count == 1 for count in ticket_counts.values())
