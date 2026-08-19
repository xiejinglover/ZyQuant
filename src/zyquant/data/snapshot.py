from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Sequence

import pandas as pd
import pyarrow.dataset as pads

from zyquant.core.exceptions import (
    DataContractError, FutureDataError, SchemaVersionError,
)
from zyquant.core.hashing import hash_file
from zyquant.core.versioning import SNAPSHOT_SCHEMA_VERSION

from .contracts import DYNAMIC_TABLES, FIELD_SPECS, TABLES, VISIBILITY_FIELDS
from .manifest import SnapshotManifest


@dataclass(frozen=True)
class SnapshotMetadata:
    dataset_id: str
    schema_version: str
    as_of_date: date
    fingerprint: str
    adjustment_version: str
    manifest_schema_version: str


class ResearchDataView:
    price_kind = "post_adjusted"

    def __init__(self, snapshot: "DataSnapshot", cutoff: date):
        self.snapshot, self.cutoff = snapshot, cutoff

    def bars(
        self, start: date, end: date,
        instruments: Sequence[str] | None = None,
        fields: Sequence[str] | None = None,
    ) -> pd.DataFrame:
        return self.snapshot.post_adjusted_bars(
            start, end, instruments, fields, self.cutoff
        )

    def table(self, name: str, **kwargs) -> pd.DataFrame:
        if name == "daily_raw":
            raise DataContractError("research view cannot access raw execution prices")
        return self.snapshot.table(name, cutoff=self.cutoff, **kwargs)


class TradingDataView:
    price_kind = "raw"

    def __init__(self, snapshot: "DataSnapshot", cutoff: date):
        self.snapshot, self.cutoff = snapshot, cutoff

    def bars(
        self, start: date, end: date,
        instruments: Sequence[str] | None = None,
        fields: Sequence[str] | None = None,
    ) -> pd.DataFrame:
        return self.snapshot.raw_bars(start, end, instruments, fields, self.cutoff)

    def table(self, name: str, **kwargs) -> pd.DataFrame:
        if name == "daily_post_adjusted":
            raise DataContractError("trading view cannot access adjusted research prices")
        return self.snapshot.table(name, cutoff=self.cutoff, **kwargs)


class FinancialDataView:
    """Point-in-time access to optional financial snapshot tables."""

    def __init__(self, snapshot: "DataSnapshot", cutoff: date):
        self.snapshot, self.cutoff = snapshot, cutoff

    def facts(
        self,
        start_period: date | None = None,
        end_period: date | None = None,
        instruments: Sequence[str] | None = None,
        item_codes: Sequence[str] | None = None,
    ) -> pd.DataFrame:
        frame = self.snapshot.table(
            "financial_facts", instruments=instruments, cutoff=self.cutoff
        )
        if start_period is not None:
            frame = frame[frame["fiscal_period_end"] >= start_period]
        if end_period is not None:
            frame = frame[frame["fiscal_period_end"] <= end_period]
        if item_codes is not None:
            frame = frame[frame["item_code"].isin(map(str, item_codes))]
        return frame.reset_index(drop=True)

    def latest_metrics(
        self,
        as_of: date,
        instruments: Sequence[str] | None = None,
        metric_codes: Sequence[str] | None = None,
        bases: Sequence[str] | None = None,
    ) -> pd.DataFrame:
        self._guard_as_of(as_of)
        frame = self.snapshot.table(
            "fundamental_metrics",
            instruments=instruments,
            cutoff=min(as_of, self.cutoff),
        )
        frame = frame[
            (frame["available_at"] <= as_of)
            & (frame["fiscal_period_end"] <= as_of)
        ]
        if metric_codes is not None:
            frame = frame[frame["metric_code"].isin(map(str, metric_codes))]
        if bases is not None:
            frame = frame[frame["basis"].isin(map(str, bases))]
        if frame.empty:
            return frame.reset_index(drop=True)
        ordered = frame.sort_values([
            "instrument_id", "metric_code", "basis",
            "fiscal_period_end", "available_at", "metric_id",
        ])
        return ordered.groupby(
            ["instrument_id", "metric_code", "basis"],
            as_index=False,
            sort=True,
        ).tail(1).reset_index(drop=True)

    def _as_of_steps(
        self,
        dates: Sequence[date],
        instruments: Sequence[str] | None,
        metric_codes: Sequence[str] | None,
        bases: Sequence[str] | None,
    ) -> tuple[pd.DataFrame, list[date]]:
        """Read the metric table once and reduce it to per-group step functions.

        `latest_metrics` selects, per `(instrument, metric_code, basis)`, the row
        maximising `(fiscal_period_end, available_at, metric_id)` among rows
        visible at the as-of date. Repeating that per date costs one full table
        scan each time, which is why `metric_panel` used to be unusable for more
        than a handful of dates.

        The same answer is a step function of the as-of date: a row becomes
        eligible on `max(available_at, fiscal_period_end)` and, once eligible,
        stays eligible. So sorting by eligibility and taking a running maximum
        of the ordering key gives, at every eligibility boundary, the row
        `latest_metrics` would have chosen — and any as-of date between two
        boundaries inherits the earlier one. Callers then forward-fill onto
        whatever date grid they need.

        Returns the boundaries frame (one row per group per boundary, carrying
        the winning row's columns) and the sorted, de-duplicated date list.
        """
        wanted = sorted({day for day in dates})
        if not wanted:
            return pd.DataFrame(), []
        for day in wanted:
            self._guard_as_of(day)
        frame = self.snapshot.table(
            "fundamental_metrics",
            instruments=instruments,
            cutoff=min(wanted[-1], self.cutoff),
        )
        if metric_codes is not None:
            frame = frame[frame["metric_code"].isin(map(str, metric_codes))]
        if bases is not None:
            frame = frame[frame["basis"].isin(map(str, bases))]
        if frame.empty:
            return frame.reset_index(drop=True), wanted

        frame = frame.reset_index(drop=True)
        available = pd.to_datetime(frame["available_at"])
        period = pd.to_datetime(frame["fiscal_period_end"])
        # A row can only be used once BOTH its period has ended and it has been
        # published; `latest_metrics` tests the two separately, and taking the
        # later of the two reproduces that without assuming their order.
        eligible = available.where(available >= period, period)

        # Rank rows by the ordering key so a running maximum over an integer
        # reproduces the lexicographic "latest report wins" comparison. Note
        # fiscal_period_end leads available_at: a late-published restatement of
        # an older period must NOT displace a newer period already on file.
        order = frame.assign(
            _period=period, _available=available,
        ).sort_values(
            ["_period", "_available", "metric_id"], kind="mergesort",
        ).index.to_numpy()
        rank = pd.Series(range(len(order)), index=order).sort_index()
        row_of_rank = pd.Series(order, index=range(len(order)))

        work = frame.assign(_eligible=eligible, _rank=rank.to_numpy())
        work = work.sort_values(
            ["instrument_id", "metric_code", "basis", "_eligible", "_rank"],
            kind="mergesort",
        )
        keys = ["instrument_id", "metric_code", "basis"]
        work["_best"] = work.groupby(keys, sort=False)["_rank"].cummax()
        # One boundary per (group, eligibility date): the last row on that date
        # already carries the running maximum for the whole date.
        work = work.drop_duplicates(keys + ["_eligible"], keep="last")

        winners = frame.loc[row_of_rank.loc[work["_best"]].to_numpy()]
        boundaries = winners.reset_index(drop=True)
        boundaries["_eligible"] = work["_eligible"].to_numpy()
        return boundaries, wanted

    def metric_panel(
        self,
        dates: Sequence[date],
        instruments: Sequence[str] | None = None,
        metric_codes: Sequence[str] | None = None,
        bases: Sequence[str] | None = None,
    ) -> pd.DataFrame:
        """Long panel of the latest visible metric per date.

        Output is one row per `(as_of_date, instrument, metric_code, basis)`, so
        it grows as dates x groups. For a dense daily panel over the whole
        history prefer `metric_matrix`, which returns the same information in
        wide form without materialising that product.
        """
        boundaries, wanted = self._as_of_steps(
            dates, instruments, metric_codes, bases
        )
        if boundaries.empty:
            return boundaries.drop(columns=["_eligible"], errors="ignore")
        columns = [name for name in boundaries.columns if name != "_eligible"]
        stamps = pd.DatetimeIndex(pd.to_datetime(pd.Series(wanted)))
        outputs = []
        keys = ["instrument_id", "metric_code", "basis"]
        for day, stamp in zip(wanted, stamps):
            visible = boundaries[boundaries["_eligible"] <= stamp]
            if visible.empty:
                continue
            current = visible.drop_duplicates(keys, keep="last")[columns].copy()
            current.insert(0, "as_of_date", day)
            outputs.append(current)
        if not outputs:
            return pd.DataFrame(columns=["as_of_date", *columns])
        return pd.concat(outputs, ignore_index=True, sort=False)

    def metric_matrix(
        self,
        dates: Sequence[date],
        metric_code: str,
        basis: str,
        instruments: Sequence[str] | None = None,
    ) -> pd.DataFrame:
        """Wide point-in-time panel for one metric: rows are dates, columns are
        instruments, each cell the value `latest_metrics` would return that day.

        This is the shape a daily factor needs. The long form of the same data
        would be dates x instruments rows — around a hundred million for a full
        A-share history — which is why it gets its own accessor.
        """
        boundaries, wanted = self._as_of_steps(
            dates, instruments, [metric_code], [basis]
        )
        index = pd.DatetimeIndex(pd.to_datetime(pd.Series(wanted)))
        if boundaries.empty:
            return pd.DataFrame(
                index=pd.Index(wanted, name="trade_date"),
                columns=pd.Index([], name="instrument_id"),
                dtype=float,
            )
        steps = boundaries.pivot_table(
            index="_eligible", columns="instrument_id", values="value",
            aggfunc="last",
        ).sort_index()
        # Union the boundaries into the requested grid, forward-fill each
        # column's step function, then keep only the requested dates.
        merged = steps.reindex(steps.index.union(index)).ffill()
        result = merged.reindex(index)
        result.index = pd.Index(wanted, name="trade_date")
        return result

    def valuation(
        self,
        start: date,
        end: date,
        instruments: Sequence[str] | None = None,
        fields: Sequence[str] | None = None,
    ) -> pd.DataFrame:
        self._guard_as_of(end)
        return self.snapshot.table(
            "daily_valuation", start, end, instruments,
            cutoff=self.cutoff, fields=fields,
        )

    def share_capital(
        self,
        as_of: date,
        instruments: Sequence[str] | None = None,
    ) -> pd.DataFrame:
        self._guard_as_of(as_of)
        frame = self.snapshot.table(
            "share_capital",
            instruments=instruments,
            cutoff=min(as_of, self.cutoff),
        )
        frame = frame[
            (frame["effective_from"] <= as_of)
            & (frame["available_at"] <= as_of)
        ]
        if frame.empty:
            return frame.reset_index(drop=True)
        return frame.sort_values([
            "instrument_id", "effective_from", "available_at",
            "capital_event_id",
        ]).groupby(
            "instrument_id", as_index=False, sort=True
        ).tail(1).reset_index(drop=True)

    def _guard_as_of(self, as_of: date) -> None:
        if as_of > self.cutoff:
            raise FutureDataError(
                f"financial as-of date {as_of} exceeds cutoff {self.cutoff}"
            )


class DataSnapshot:
    def __init__(self, path: str | Path, verify_hashes: bool = True):
        self.path = Path(path).expanduser().resolve()
        manifest_path = self.path / "manifest.json"
        if not manifest_path.exists():
            raise DataContractError(f"missing snapshot manifest: {manifest_path}")
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        try:
            model = SnapshotManifest.model_validate(payload)
        except Exception as exc:
            raise SchemaVersionError(
                "snapshot is not a ZyQuant v1 manifest; republish the dataset"
            ) from exc
        if model.manifest_schema_version != SNAPSHOT_SCHEMA_VERSION:
            raise SchemaVersionError(
                f"snapshot schema {model.manifest_schema_version} is not supported"
            )
        if not model.adjustment.materialized:
            raise DataContractError("snapshot requires materialized post-adjusted bars")
        self.manifest_model = model
        self.manifest = payload
        self._manifest_tables = {table.name for table in model.tables}
        self.metadata = SnapshotMetadata(
            dataset_id=model.dataset_id,
            schema_version=model.schema_version,
            as_of_date=model.as_of_date,
            fingerprint=model.fingerprint,
            adjustment_version=model.adjustment.algorithm_version,
            manifest_schema_version=model.manifest_schema_version,
        )
        if verify_hashes:
            for info in model.files:
                file_path = self.path / info.path
                if not file_path.exists() or hash_file(file_path) != info.sha256:
                    raise DataContractError(f"snapshot file hash mismatch: {file_path}")

    def research(self, cutoff: date) -> ResearchDataView:
        return ResearchDataView(self, cutoff)

    def trading(self, cutoff: date) -> TradingDataView:
        return TradingDataView(self, cutoff)

    def financial(self, cutoff: date) -> FinancialDataView:
        capabilities = self.manifest.get("capabilities", {})
        if not isinstance(capabilities, dict) or "financials" not in capabilities:
            raise DataContractError(
                "snapshot does not provide the optional financial capability"
            )
        return FinancialDataView(self, cutoff)

    def raw_bars(
        self,
        start: date,
        end: date,
        instruments: Sequence[str] | None = None,
        fields: Sequence[str] | None = None,
        cutoff: date | datetime | None = None,
    ) -> pd.DataFrame:
        return self._bars("daily_raw", start, end, instruments, fields, cutoff)

    def post_adjusted_bars(
        self,
        start: date,
        end: date,
        instruments: Sequence[str] | None = None,
        fields: Sequence[str] | None = None,
        cutoff: date | datetime | None = None,
    ) -> pd.DataFrame:
        return self._bars("daily_post_adjusted", start, end, instruments, fields, cutoff)

    def table(
        self,
        name: str,
        start: date | None = None,
        end: date | None = None,
        instruments: Sequence[str] | None = None,
        cutoff: date | datetime | None = None,
        fields: Sequence[str] | None = None,
        dates: Sequence[date] | None = None,
    ) -> pd.DataFrame:
        if name not in TABLES:
            raise DataContractError(f"unknown canonical table: {name}")
        if name in DYNAMIC_TABLES and cutoff is None:
            raise FutureDataError(f"{name} requires an explicit cutoff")
        self._guard(end, cutoff)
        if name not in self._manifest_tables:
            raise DataContractError(
                f"snapshot manifest does not contain table: {name}"
            )
        path = self.path / name
        if not path.exists():
            raise DataContractError(f"snapshot table does not exist: {name}")
        dataset = pads.dataset(path, format="parquet", partitioning="hive")
        schema_names = set(dataset.schema.names)
        predicate = None
        date_column = "trade_date" if "trade_date" in schema_names else None
        if date_column and start is not None:
            predicate = pads.field(date_column) >= start
        if date_column and end is not None:
            current = pads.field(date_column) <= end
            predicate = current if predicate is None else predicate & current
        if dates is not None:
            if date_column is None:
                raise DataContractError(
                    f"{name} does not support trade-date filtering"
                )
            wanted_dates = sorted(set(dates))
            current = pads.field(date_column).isin(wanted_dates)
            predicate = current if predicate is None else predicate & current
        if instruments is not None and "instrument_id" in schema_names:
            current = pads.field("instrument_id").isin([str(item) for item in instruments])
            predicate = current if predicate is None else predicate & current
        cutoff_date = cutoff.date() if isinstance(cutoff, datetime) else cutoff
        visibility = VISIBILITY_FIELDS.get(name)
        if visibility is not None and visibility not in schema_names:
            raise DataContractError(
                f"{name} is missing its PIT visibility field: {visibility}"
            )
        if (
            cutoff_date is not None
            and visibility is not None
            and visibility in schema_names
        ):
            current = pads.field(visibility).is_null() | (
                pads.field(visibility) <= cutoff_date
            )
            predicate = current if predicate is None else predicate & current
        requested = list(fields) if fields is not None else None
        required = [
            column for column in ("trade_date", "instrument_id")
            if column in schema_names
        ]
        if requested is not None:
            unknown_fields = set(requested) - set(FIELD_SPECS[name])
            if unknown_fields:
                raise DataContractError(
                    f"{name} has unknown requested fields: "
                    f"{sorted(unknown_fields)}"
                )
            selected = list(dict.fromkeys(required + requested))
            missing = set(selected) - schema_names
            if missing:
                raise DataContractError(
                    f"{name} missing requested fields: {sorted(missing)}"
                )
        else:
            selected = None
        frame = dataset.to_table(filter=predicate, columns=selected).to_pandas()
        for column in (
            "trade_date", "list_date", "delist_date", "record_date", "ex_date",
            "pay_date", "announced_at", "effective_from", "effective_to", "known_at",
            "fiscal_period_start", "fiscal_period_end", "filing_period_end",
            "published_at", "available_at",
        ):
            if column in frame:
                converted = pd.to_datetime(frame[column], errors="coerce")
                frame[column] = converted.map(
                    lambda value: value.date() if pd.notna(value) else None
                ).astype(object)
        sort_columns = [
            column for column in ("trade_date", "instrument_id", "effective_from")
            if column in frame
        ]
        if sort_columns:
            frame.sort_values(sort_columns, inplace=True, ignore_index=True)
        return frame.reset_index(drop=True)

    def market_rule(self, day: date, exchange: str, asset_type: str):
        rules = self.table("market_rules", cutoff=day)
        current = rules[
            (rules["exchange"].astype(str) == str(exchange))
            & (rules["asset_type"].astype(str) == str(asset_type))
            & (rules["effective_from"] <= day)
            & (rules["effective_to"].isna() | (rules["effective_to"] >= day))
        ]
        if len(current) != 1:
            raise DataContractError(
                f"expected one market rule for {exchange}/{asset_type} on {day}, "
                f"found {len(current)}"
            )
        return current.iloc[0]

    def _bars(self, name, start, end, instruments, fields, cutoff):
        if cutoff is None:
            raise FutureDataError(f"{name} requires an explicit cutoff")
        self._guard_price_coverage(instruments)
        requested = None
        if fields is not None:
            requested = [
                field for field in fields
                if field not in {"trade_date", "instrument_id"}
            ]
        return self.table(
            name, start, end, instruments, cutoff, requested
        )

    def _guard_price_coverage(
        self,
        instruments: Sequence[str] | None,
    ) -> None:
        if instruments is None:
            return
        lineage = self.manifest.get("lineage", {})
        coverage = (
            lineage.get("coverage", {})
            if isinstance(lineage, dict)
            else {}
        )
        prices = (
            coverage.get("prices", {})
            if isinstance(coverage, dict)
            else {}
        )
        declared = prices.get("instruments") if isinstance(prices, dict) else None
        if not isinstance(declared, list):
            return
        missing = sorted(set(map(str, instruments)) - set(map(str, declared)))
        if missing:
            raise DataContractError(
                "snapshot price coverage does not include requested instruments: "
                f"{missing[:10]}"
            )

    @staticmethod
    def _guard(end: date | None, cutoff: date | datetime | None) -> None:
        if end is None or cutoff is None:
            return
        cutoff_date = cutoff.date() if isinstance(cutoff, datetime) else cutoff
        if end > cutoff_date:
            raise FutureDataError(f"requested end {end} exceeds cutoff {cutoff_date}")
