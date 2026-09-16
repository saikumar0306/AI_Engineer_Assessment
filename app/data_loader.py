from pathlib import Path

import pandas as pd


EXPECTED_COLUMNS = [
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
]

NUMERIC_COLUMNS = [
    "response_time_hrs",
    "resolution_time_hrs",
    "customer_rating",
]

ALLOWED_VALUES = {
    "category": {"Billing", "Technical", "General"},
    "priority": {"Low", "Medium", "High", "Critical"},
    "status": {"Open", "Resolved", "Escalated"},
}

DEFAULT_DATA_PATH = Path(__file__).parent.parent / "data" / "support_tickets.csv"


def validate_tickets(tickets: pd.DataFrame) -> None:
    """Validate the schema and basic values of a loaded ticket DataFrame."""
    if tickets.empty:
        raise ValueError("The ticket dataset is empty.")

    if list(tickets.columns) != EXPECTED_COLUMNS:
        raise ValueError(
            "The dataset must contain exactly these columns: "
            f"{', '.join(EXPECTED_COLUMNS)}"
        )

    if tickets["ticket_id"].isna().any() or tickets["ticket_id"].duplicated().any():
        raise ValueError("ticket_id values must be present and unique.")

    if tickets["created_at"].isna().any():
        raise ValueError("created_at contains invalid or missing values.")

    for column in NUMERIC_COLUMNS:
        if tickets[column].isna().any() and column == "response_time_hrs":
            raise ValueError("response_time_hrs cannot contain missing values.")
        if (tickets[column].dropna() < 0).any():
            raise ValueError(f"{column} cannot contain negative values.")

    for column, allowed_values in ALLOWED_VALUES.items():
        values = set(tickets[column].dropna().unique())
        if not values.issubset(allowed_values):
            invalid_values = sorted(values - allowed_values)
            raise ValueError(f"Invalid values in {column}: {invalid_values}")


def load_tickets(path: str | Path = DEFAULT_DATA_PATH) -> pd.DataFrame:
    """Load, normalize, and validate support tickets from a CSV file."""
    csv_path = Path(path)
    if not csv_path.is_file():
        raise FileNotFoundError(f"Ticket dataset not found: {csv_path}")

    tickets = pd.read_csv(csv_path)

    if set(tickets.columns) != set(EXPECTED_COLUMNS):
        raise ValueError(
            "The dataset must contain exactly these columns: "
            f"{', '.join(EXPECTED_COLUMNS)}"
        )
    tickets = tickets[EXPECTED_COLUMNS]

    tickets["created_at"] = pd.to_datetime(
        tickets["created_at"], format="%Y-%m-%d %H:%M", errors="coerce"
    )
    for column in NUMERIC_COLUMNS:
        tickets[column] = pd.to_numeric(tickets[column], errors="coerce")

    validate_tickets(tickets)
    return tickets