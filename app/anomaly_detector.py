from __future__ import annotations

from typing import Any

import pandas as pd

from app.data_loader import load_tickets


class AnomalyDetector:
    """Detects explainable anomalies in support ticket data."""

    def __init__(self, data_path: str | None = None):
        self.tickets = load_tickets(data_path) if data_path is not None else load_tickets()

    def calculate_resolution_iqr(self) -> tuple[float, float, float, float]:
        """Return Q1, Q3, IQR and upper fence for resolution_time_hrs."""
        valid = self.tickets["resolution_time_hrs"].dropna()
        if valid.empty:
            raise ValueError("No valid resolution_time_hrs values available for IQR calculation.")

        q1 = float(valid.quantile(0.25))
        q3 = float(valid.quantile(0.75))
        iqr = q3 - q1
        upper_fence = q3 + 1.5 * iqr
        return q1, q3, iqr, upper_fence

    def find_long_resolution_time_anomalies(self) -> list[dict[str, Any]]:
        """Find tickets whose resolution time exceeds the IQR-based upper fence."""
        q1, q3, iqr, upper_fence = self.calculate_resolution_iqr()
        valid = self.tickets[self.tickets["resolution_time_hrs"].notna()].copy()
        flagged = valid[valid["resolution_time_hrs"] > upper_fence].copy()

        anomalies: list[dict[str, Any]] = []
        for _, row in flagged.drop_duplicates(subset=["ticket_id"]).iterrows():
            anomalies.append(
                {
                    "ticket_id": row["ticket_id"],
                    "anomaly_type": "long_resolution_time",
                    "reason": (
                        f"resolution_time_hrs={row['resolution_time_hrs']} exceeds upper fence "
                        f"Q3 + 1.5*IQR = {upper_fence:.2f} (Q1={q1:.2f}, Q3={q3:.2f}, IQR={iqr:.2f})"
                    ),
                    "resolution_time_hrs": float(row["resolution_time_hrs"]),
                }
            )
        return anomalies

    def find_high_priority_unresolved_anomalies(
        self,
        reference_time: pd.Timestamp | str | None = None,
    ) -> list[dict[str, Any]]:
        """Find unresolved high/critical tickets older than 24 hours."""
        if reference_time is None:
            reference_time = pd.Timestamp.now(tz=None)
        else:
            reference_time = pd.Timestamp(reference_time)

        candidates = self.tickets[
            self.tickets["priority"].isin(["High", "Critical"])
            & self.tickets["status"].isin(["Open", "Escalated"])
            & self.tickets["created_at"].notna()
        ].copy()

        flagged = candidates[
            (reference_time - candidates["created_at"]).dt.total_seconds() / 3600 > 24
        ].copy()

        anomalies: list[dict[str, Any]] = []
        for _, row in flagged.drop_duplicates(subset=["ticket_id"]).iterrows():
            age_hours = (reference_time - row["created_at"]).total_seconds() / 3600
            anomalies.append(
                {
                    "ticket_id": row["ticket_id"],
                    "anomaly_type": "high_priority_unresolved",
                    "reason": (
                        f"priority={row['priority']} and status={row['status']} unresolved for "
                        f"{age_hours:.2f} hours (>24h)"
                    ),
                    "priority": row["priority"],
                    "status": row["status"],
                    "created_at": row["created_at"],
                }
            )
        return anomalies

    def find_all_anomalies(
        self,
        reference_time: pd.Timestamp | str | None = None,
    ) -> list[dict[str, Any]]:
        """Return all explainable anomalies without duplicate records for the same rule."""
        long_resolution = self.find_long_resolution_time_anomalies()
        high_priority = self.find_high_priority_unresolved_anomalies(reference_time=reference_time)
        return long_resolution + high_priority


__all__ = ["AnomalyDetector"]
