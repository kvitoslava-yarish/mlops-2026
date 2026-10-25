import re
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

MONTH_RE = re.compile(r"^\d{4}-(0[1-9]|1[0-2])$")


@dataclass(frozen=True)
class Batch:
    month: str  # "YYYY-MM"
    start: datetime  # inclusive, UTC
    end: datetime  # exclusive, UTC

    @property
    def hours(self) -> int:
        return int((self.end - self.start) / timedelta(hours=1))

    @property
    def previous(self) -> "Batch":
        return batch_for((self.start - timedelta(days=1)).strftime("%Y-%m"))


def batch_for(month: str) -> Batch:
    if not MONTH_RE.match(month):
        raise ValueError(f"batch month must look like YYYY-MM, got {month!r}")
    start = datetime.strptime(month, "%Y-%m").replace(tzinfo=UTC)
    end = (start + timedelta(days=32)).replace(day=1)
    return Batch(month=month, start=start, end=end)


def resolve_batch(batch_month: str | None, data_interval_start: datetime | None) -> Batch:
    """An explicit ``batch_month`` parameter wins; otherwise the month of the data interval."""
    if batch_month:
        return batch_for(batch_month)
    if data_interval_start is None:
        raise ValueError("no batch: pass batch_month='YYYY-MM' or run with a data interval")
    return batch_for(data_interval_start.astimezone(UTC).strftime("%Y-%m"))


def month_range(first: str, last: str) -> list[str]:
    months, b = [], batch_for(first)
    while b.month <= last:
        months.append(b.month)
        b = batch_for(b.end.strftime("%Y-%m"))
    return months
