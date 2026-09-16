from __future__ import annotations

from typing import Any

import pandas as pd

from app.data_loader import load_tickets


class QueryEngine:
    """Deterministic query layer for the support ticket dataset."""

    def __init__(self, data_path: str | None = None):
        self.tickets = load_tickets(data_path) if data_path is not None else load_tickets()

    def count_tickets(self, **filters: Any) -> int:
        """Return the number of tickets matching optional filters."""
        data = self.filter_tickets(**filters)
        return int(len(data))

    def filter_tickets(self, **filters: Any) -> pd.DataFrame:
        """Apply equality and multi-value filters to the ticket dataset."""
        data = self.tickets.copy()
        for key, value in filters.items():
            if value is None:
                continue
            if key not in data.columns:
                raise ValueError(f"Unsupported filter '{key}'.")
            if isinstance(value, (list, tuple, set, pd.Index)):
                values = list(value)
                data = data[data[key].isin(values)]
            else:
                data = data[data[key] == value]
        return data

    def average_response_time(self, **filters: Any) -> float:
        data = self.filter_tickets(**filters)
        return float(data["response_time_hrs"].mean())

    def average_resolution_time(self, **filters: Any) -> float:
        data = self.filter_tickets(**filters)
        return float(data["resolution_time_hrs"].mean())

    def average_customer_rating(self, **filters: Any) -> float:
        data = self.filter_tickets(**filters)
        return float(data["customer_rating"].mean())

    def group_by_category(self) -> pd.Series:
        return self.tickets["category"].value_counts().sort_values(ascending=False)

    def group_by_priority(self) -> pd.Series:
        return self.tickets["priority"].value_counts().sort_values(ascending=False)

    def group_by_status(self) -> pd.Series:
        return self.tickets["status"].value_counts().sort_values(ascending=False)

    def group_by_agent(self) -> pd.Series:
        return self.tickets["agent_id"].value_counts().sort_values(ascending=False)

    def find_unresolved_tickets(self) -> pd.DataFrame:
        return self.filter_tickets(status=["Open", "Escalated"])

    def find_critical_or_high_unresolved(self) -> pd.DataFrame:
        return self.tickets[
            self.tickets["status"].isin(["Open", "Escalated"])
            & self.tickets["priority"].isin(["Critical", "High"])
        ].copy()

    def find_tickets_with_resolution_time_above(self, threshold: float) -> pd.DataFrame:
        return self.tickets[self.tickets["resolution_time_hrs"] > threshold].copy()

    def agent_with_most_resolved_tickets(self) -> str:
        resolved = self.tickets[self.tickets["status"].eq("Resolved")].copy()
        top_agent = resolved["agent_id"].value_counts().idxmax()
        return str(top_agent)

    def describe(self) -> dict[str, Any]:
        return {
            "count": len(self.tickets),
            "average_response_time_hrs": self.average_response_time(),
            "average_resolution_time_hrs": self.average_resolution_time(),
            "average_customer_rating": self.average_customer_rating(),
            "by_category": self.group_by_category().to_dict(),
            "by_priority": self.group_by_priority().to_dict(),
            "by_status": self.group_by_status().to_dict(),
            "agent_with_most_resolved_tickets": self.agent_with_most_resolved_tickets(),
        }
