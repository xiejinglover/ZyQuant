from __future__ import annotations

from dataclasses import dataclass
from datetime import date


class DailySchedule:
    def decision_dates(self, calendar: list[date]) -> list[date]:
        return list(calendar)


@dataclass(frozen=True)
class ExplicitDateSchedule:
    dates: tuple[date, ...]

    def decision_dates(self, calendar: list[date]) -> list[date]:
        available = set(calendar)
        missing = set(self.dates) - available
        if missing:
            raise ValueError(f"explicit schedule contains non-trading dates: {sorted(missing)}")
        return sorted(set(self.dates))


@dataclass(frozen=True)
class EveryNTradingDays:
    n: int
    offset: int = 0

    def decision_dates(self, calendar: list[date]) -> list[date]:
        if self.n < 1:
            raise ValueError("n must be positive")
        return list(calendar[self.offset::self.n])


@dataclass(frozen=True)
class WeeklySchedule:
    position: str = "last"

    def decision_dates(self, calendar: list[date]) -> list[date]:
        if self.position not in {"first", "last"}:
            raise ValueError("weekly position must be first or last")
        groups: dict[tuple[int, int], list[date]] = {}
        for day in calendar:
            iso = day.isocalendar()
            groups.setdefault((iso.year, iso.week), []).append(day)
        return [values[0 if self.position == "first" else -1] for values in groups.values()]


@dataclass(frozen=True)
class MonthlySchedule:
    position: str = "last"

    def decision_dates(self, calendar: list[date]) -> list[date]:
        if self.position not in {"first", "last"}:
            raise ValueError("monthly position must be first or last")
        groups: dict[tuple[int, int], list[date]] = {}
        for day in calendar:
            groups.setdefault((day.year, day.month), []).append(day)
        return [values[0 if self.position == "first" else -1] for values in groups.values()]
