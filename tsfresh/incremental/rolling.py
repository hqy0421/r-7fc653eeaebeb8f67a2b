# -*- coding: utf-8 -*-
"""
Incremental rolling feature table.

The :class:`IncrementalRollingFeatureTable` maintains native tsfresh features
over fixed, left-closed/right-open rolling windows while the underlying
event stream is backfilled, corrected or deleted.

Design summary
==============

* Samples are fixed up front: each sample declares an id, an entity, a
  cutoff time and a lookback length.  Its window is
  ``[cutoff - lookback, cutoff)`` -- left-closed, right-open.
* Events carry a stable ``event_id`` and a monotonic ``revision``.  They
  arrive as upserts (insert/correction) or deletes.  The raw event log is
  the authority and is stored whole, every generation.
* Revisions are last-writer-write per event id: older revisions are
  skipped as stale, exact repeats are skipped as duplicates and reusing a
  revision number with a different payload (or contradicting a delete)
  raises :class:`RevisionConflictError` -- nothing is applied.
* After each batch the affected windows are identified from both the old
  and the new location of every migrated event.  A window is handed back
  to the *native* :func:`tsfresh.extract_features` only when its actual
  input content changed (content signature comparison); unchanged rows are
  preserved byte-for-byte.  Empty windows and missing kinds keep their
  declared columns (filled with ``NaN``); features are never computed by
  hand.
* Every successful batch produces a new immutable generation containing
  the raw authority, the native feature configuration, the samples, the
  real feature matrix and the window provenance.  Writes are atomic: on
  failure the previous generation stays authoritative.
"""

import datetime
import hashlib
import math
from collections import defaultdict
from dataclasses import dataclass

import numpy as np
import pandas as pd

from tsfresh.feature_extraction.extraction import extract_features
from tsfresh.incremental.store import GenerationStore, StoreError

__all__ = [
    "IncrementalRollingFeatureTable",
    "UpdateReport",
    "WindowSource",
    "Sample",
    "IncrementalError",
    "EventValidationError",
    "RevisionConflictError",
    "StoreError",
]

# ---------------------------------------------------------------------- #
# Canonical column names.  The double-underscore prefix keeps them distinct
# from user-supplied kind names when frames are passed to the native
# extractor.
# ---------------------------------------------------------------------- #
EVENT_ID = "event_id"
REVISION = "revision"
OP = "op"
ENTITY = "entity"
TIME = "time"
KIND = "kind"
VALUE = "value"

SAMPLE_ID = "sample_id"
CUTOFF = "cutoff"
LOOKBACK = "lookback_length"

_S_ID = "__inc_sample_id"
_S_SORT = "__inc_time"
_S_KIND = "__inc_kind"
_S_VALUE = "__inc_value"

_OP_UPSERT = "upsert"
_OP_DELETE = "delete"
_OP_ALIASES = {
    "upsert": _OP_UPSERT,
    "insert": _OP_UPSERT,
    "update": _OP_UPSERT,
    "correct": _OP_UPSERT,
    "correction": _OP_UPSERT,
    "delete": _OP_DELETE,
    "remove": _OP_DELETE,
    "tombstone": _OP_DELETE,
}


class IncrementalError(RuntimeError):
    """Base class for all errors raised by the incremental feature table."""


class EventValidationError(IncrementalError, ValueError):
    """Raised when an event batch or sample declaration is malformed."""


class RevisionConflictError(IncrementalError):
    """
    Raised when an event reuses a known revision number with a payload that
    contradicts the revision already recorded for that event id, or applies
    an operation that contradicts the recorded state (e.g. deleting an
    already-deleted revision).  Nothing from the offending batch is
    applied.
    """


@dataclass(frozen=True)
class Sample:
    """A fixed rolling-window declaration."""

    sample_id: object
    entity: object
    cutoff: object
    lookback_length: object
    window_start: object


@dataclass(frozen=True)
class WindowSource:
    """
    Provenance of one sample's feature row: the exact window boundaries,
    the stable event ids that made up the window when it was last built
    and the content signature used to decide whether recomputation is
    necessary.
    """

    sample_id: object
    entity: object
    window_start: object
    window_end_exclusive: object
    event_ids: tuple
    content_signature: str
    last_extracted_generation: object


@dataclass(frozen=True)
class UpdateReport:
    """Summary of what one :meth:`apply_events` call did."""

    generation: int
    received: int
    effective: int
    stale_skipped: int
    duplicate_skipped: int
    affected_samples: tuple
    recomputed_samples: tuple
    unchanged_samples: tuple
    migrated_events: tuple
    committed: bool


# ---------------------------------------------------------------------- #
# Small validation / canonicalization helpers
# ---------------------------------------------------------------------- #
def _is_real_number(value):
    return isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(
        value, bool
    )


def _is_datetime_like(value):
    return isinstance(
        value, (pd.Timestamp, datetime.datetime, datetime.date, np.datetime64)
    )


def _normalize_time(value):
    if isinstance(value, np.datetime64):
        return pd.Timestamp(value)
    if isinstance(value, datetime.datetime):
        return pd.Timestamp(value)
    if isinstance(value, datetime.date):
        return pd.Timestamp(value)
    return value


def _time_kind(value):
    return "datetime" if _is_datetime_like(value) else "numeric"


def _canonical_time(value):
    if isinstance(value, pd.Timestamp):
        return "ts:" + value.isoformat()
    if isinstance(value, np.datetime64):
        return "ts:" + pd.Timestamp(value).isoformat()
    if isinstance(value, (int, np.integer)):
        return f"int:{int(value)}"
    return f"float:{float(value)!r}"


def _tagged(value):
    return f"{type(value).__name__}:{value!r}"


def _hashable(value):
    try:
        hash(value)
    except TypeError as exc:
        raise EventValidationError(f"unhashable value: {value!r}") from exc
    return value


# ---------------------------------------------------------------------- #
# Main class
# ---------------------------------------------------------------------- #
class IncrementalRollingFeatureTable:
    """
    An incrementally maintained table of native tsfresh rolling features.

    :param samples: DataFrame declaring the fixed samples.  It must contain
        the columns ``sample_id``, ``entity``, ``cutoff`` and
        ``lookback_length`` and never changes for the life of the table
        (use :meth:`load` to reopen a persisted table).
    :param default_fc_parameters: native tsfresh feature calculator
        configuration, identical in meaning to the ``default_fc_parameters``
        argument of :func:`tsfresh.extract_features`.
    :param kind_to_fc_parameters: per-kind native configuration, identical
        in meaning to the ``kind_to_fc_parameters`` argument of
        :func:`tsfresh.extract_features`.
    :param initial_events: optional DataFrame of events (same schema as
        :meth:`apply_events`) used to build generation 0.
    :param store_dir: if given, every generation is persisted under this
        directory and the table can later be reopened with :meth:`load`.
        If ``None``, the table lives in memory only.
    :param keep_last: if not ``None``, prune the store to at most this many
        generations after each commit.
    """

    def __init__(
        self,
        samples,
        default_fc_parameters=None,
        kind_to_fc_parameters=None,
        initial_events=None,
        store_dir=None,
        keep_last=None,
    ):
        if not isinstance(samples, pd.DataFrame):
            raise EventValidationError("samples must be a pandas DataFrame")

        self._config = {
            "default_fc_parameters": default_fc_parameters,
            "kind_to_fc_parameters": kind_to_fc_parameters,
        }
        self._store = GenerationStore(store_dir) if store_dir is not None else None
        if keep_last is not None and (not isinstance(keep_last, int) or keep_last < 1):
            raise EventValidationError(
                f"keep_last must be a positive integer or None, got {keep_last!r}"
            )
        self._keep_last = keep_last
        self._time_kind = None  # "numeric" | "datetime", learned from events

        if self._store is not None and self._store.current_generation_name() is not None:
            raise EventValidationError(
                f"store {store_dir!r} already contains generations; "
                "use IncrementalRollingFeatureTable.load() to reopen it"
            )

        self._samples, self._sample_index, self._entity_samples = self._build_samples(
            samples
        )

        # Authority: event_id -> record dict
        # record = {entity, time, kind, value, revision, alive}
        self._authority = {}
        self._generation = -1
        self._features = pd.DataFrame(index=self._sample_index.copy(), dtype=float)
        self._provenance = {}
        self._declared_columns = []
        self._probed_kinds = set()

        if initial_events is not None:
            self.apply_events(initial_events)
        else:
            # Generation 0 with an empty authority: empty windows, declared
            # columns only (per-kind configs are probed eagerly).
            self._probe_configured_kinds()
            self._features = self._features.reindex(columns=self._declared_columns)
            for sample in self._samples:
                self._provenance[sample.sample_id] = self._empty_source(sample)
            self._generation = 0
            self._commit([])

    # ------------------------------------------------------------------ #
    # Construction / persistence
    # ------------------------------------------------------------------ #
    @classmethod
    def load(cls, store_dir):
        """Reopen a previously persisted table from ``store_dir``."""
        store = GenerationStore(store_dir)
        payload = store.load_current()

        table = cls.__new__(cls)
        table._store = store
        table._config = payload["config"]
        table._time_kind = payload["meta"].get("time_kind")
        table._keep_last = payload["meta"].get("keep_last")
        table._authority = payload["authority"]["records"]
        table._generation = payload["generation"]
        table._features = payload["features"].copy()
        table._provenance = dict(payload["provenance"])
        table._declared_columns = list(payload["features"].columns)
        table._probed_kinds = set(payload["meta"].get("probed_kinds", []))

        sample_records = payload["samples"]
        table._samples = [Sample(**record) for record in sample_records["samples"]]
        table._sample_index = pd.Index(
            [s.sample_id for s in table._samples], name=SAMPLE_ID
        )
        table._entity_samples = defaultdict(list)
        for sample in table._samples:
            table._entity_samples[sample.entity].append(sample)
        return table

    def _build_samples(self, samples):
        required = [SAMPLE_ID, ENTITY, CUTOFF, LOOKBACK]
        missing = [col for col in required if col not in samples.columns]
        if missing:
            raise EventValidationError(
                f"samples DataFrame is missing required columns: {missing}"
            )
        if samples[[SAMPLE_ID, ENTITY, CUTOFF, LOOKBACK]].isnull().any().any():
            raise EventValidationError(
                "sample_id, entity, cutoff and lookback_length must not contain NaN"
            )
        if samples[SAMPLE_ID].duplicated().any():
            raise EventValidationError("sample_id values must be unique")

        built = []
        for row in samples.itertuples(index=False):
            sample_id = _hashable(getattr(row, SAMPLE_ID))
            entity = _hashable(getattr(row, ENTITY))
            cutoff = getattr(row, CUTOFF)
            lookback = getattr(row, LOOKBACK)
            if isinstance(cutoff, np.datetime64):
                cutoff = pd.Timestamp(cutoff)
            if isinstance(lookback, np.timedelta64):
                lookback = pd.Timedelta(lookback)
            try:
                window_start = cutoff - lookback
            except TypeError as exc:
                raise EventValidationError(
                    f"incompatible cutoff/lookback pair for sample {sample_id!r}: "
                    f"{cutoff!r} - {lookback!r}"
                ) from exc
            built.append(
                Sample(
                    sample_id=sample_id,
                    entity=entity,
                    cutoff=cutoff,
                    lookback_length=lookback,
                    window_start=window_start,
                )
            )

        index = pd.Index([s.sample_id for s in built], name=SAMPLE_ID)
        entity_samples = defaultdict(list)
        for sample in built:
            entity_samples[sample.entity].append(sample)
        return built, index, entity_samples

    def _state_payload(self, recomputed, affected):
        authority_payload = {"records": self._authority}
        samples_payload = {
            "samples": [
                {
                    "sample_id": s.sample_id,
                    "entity": s.entity,
                    "cutoff": s.cutoff,
                    "lookback_length": s.lookback_length,
                    "window_start": s.window_start,
                }
                for s in self._samples
            ]
        }
        meta = {
            "n_authority_records": len(self._authority),
            "n_live_events": sum(1 for r in self._authority.values() if r["alive"]),
            "n_samples": len(self._samples),
            "n_feature_columns": len(self._declared_columns),
            "n_affected_samples": len(affected),
            "n_recomputed_samples": len(recomputed),
            "time_kind": self._time_kind,
            "probed_kinds": sorted(self._probed_kinds),
            "keep_last": self._keep_last,
        }
        return authority_payload, samples_payload, meta

    def _commit(self, recomputed, affected=()):
        if self._store is None:
            return False
        authority_payload, samples_payload, meta = self._state_payload(
            recomputed, affected
        )
        self._store.commit_generation(
            generation=self._generation,
            authority=authority_payload,
            config=self._config,
            samples=samples_payload,
            features=self._features,
            provenance=self._provenance,
            extra_meta=meta,
            keep_last=self._keep_last,
        )
        return True

    # ------------------------------------------------------------------ #
    # Event validation
    # ------------------------------------------------------------------ #
    def _validate_batch(self, events):
        if not isinstance(events, pd.DataFrame):
            raise EventValidationError("events must be a pandas DataFrame")
        required = [EVENT_ID, REVISION, OP, ENTITY, TIME, KIND, VALUE]
        missing = [col for col in required if col not in events.columns]
        if missing:
            raise EventValidationError(
                f"events DataFrame is missing required columns: {missing}"
            )
        if events[EVENT_ID].isnull().any():
            raise EventValidationError(f"{EVENT_ID} must not contain NaN")
        if events[REVISION].isnull().any():
            raise EventValidationError(f"{REVISION} must not contain NaN")
        if len(events) == 0:
            raise EventValidationError("events batch must not be empty")

        normalized = []
        batch_time_kind = None
        for row in events.itertuples(index=False):
            event_id = _hashable(getattr(row, EVENT_ID))
            revision = getattr(row, REVISION)
            if isinstance(revision, bool) or not isinstance(
                revision, (int, np.integer)
            ):
                raise EventValidationError(
                    f"revision for event {event_id!r} must be an integer, got "
                    f"{revision!r}"
                )
            revision = int(revision)
            if revision < 0:
                raise EventValidationError(
                    f"revision for event {event_id!r} must be >= 0"
                )

            raw_op = getattr(row, OP)
            if not isinstance(raw_op, str) or raw_op.lower() not in _OP_ALIASES:
                raise EventValidationError(
                    f"unknown op {raw_op!r} for event {event_id!r}; expected one of "
                    f"{sorted(set(_OP_ALIASES))}"
                )
            op = _OP_ALIASES[raw_op.lower()]

            entity = getattr(row, ENTITY)
            time_value = getattr(row, TIME)
            kind = getattr(row, KIND)
            value = getattr(row, VALUE)

            if op == _OP_UPSERT:
                if pd.isna(entity):
                    raise EventValidationError(
                        f"upsert event {event_id!r} requires a non-null entity"
                    )
                if pd.isna(time_value):
                    raise EventValidationError(
                        f"upsert event {event_id!r} requires a non-null time"
                    )
                if pd.isna(kind):
                    raise EventValidationError(
                        f"upsert event {event_id!r} requires a non-null kind"
                    )
                if pd.isna(value):
                    raise EventValidationError(
                        f"upsert event {event_id!r} requires a non-null value"
                    )
                entity = _hashable(entity)
                time_value = _normalize_time(time_value)
                if not (_is_real_number(time_value) or _is_datetime_like(time_value)):
                    raise EventValidationError(
                        f"event time must be numeric or datetime-like, got "
                        f"{time_value!r} for event {event_id!r}"
                    )
                event_time_kind = _time_kind(time_value)
                if batch_time_kind is None:
                    batch_time_kind = event_time_kind
                elif event_time_kind != batch_time_kind:
                    raise EventValidationError(
                        f"event {event_id!r} has {event_time_kind} time but another "
                        f"event in the same batch uses {batch_time_kind} time"
                    )
                if self._time_kind is not None and event_time_kind != self._time_kind:
                    raise EventValidationError(
                        f"event {event_id!r} has {event_time_kind} time but the "
                        f"table already uses {self._time_kind} times"
                    )
                self._check_time_against_samples(time_value, event_id)
                kind = str(kind)
                try:
                    value = float(value)
                except (TypeError, ValueError) as exc:
                    raise EventValidationError(
                        f"value for event {event_id!r} must be numeric, got {value!r}"
                    ) from exc
                if math.isnan(value):
                    raise EventValidationError(
                        f"value for event {event_id!r} must not be NaN"
                    )
            else:  # delete: location columns are ignored but normalized if present
                if not pd.isna(time_value):
                    time_value = _normalize_time(time_value)
                if not pd.isna(kind):
                    kind = str(kind)
                if not pd.isna(entity):
                    entity = _hashable(entity)

            normalized.append(
                {
                    EVENT_ID: event_id,
                    REVISION: revision,
                    OP: op,
                    ENTITY: entity,
                    TIME: time_value,
                    KIND: kind,
                    VALUE: value,
                }
            )

        # Duplicate event ids inside one batch must be exact repeats,
        # otherwise the batch is ambiguous.
        by_id = {}
        for record in normalized:
            eid = record[EVENT_ID]
            if eid in by_id and by_id[eid] != record:
                raise EventValidationError(
                    f"event_id {eid!r} appears multiple times in the same batch with "
                    "conflicting rows"
                )
            by_id[eid] = record

        # All checks passed: adopt the batch's time kind now.
        if self._time_kind is None and batch_time_kind is not None:
            self._time_kind = batch_time_kind
        return normalized

    def _check_time_against_samples(self, time_value, event_id):
        kind = _time_kind(time_value)
        for sample in self._samples:
            sample_kind = _time_kind(sample.cutoff)
            if sample_kind != kind:
                raise EventValidationError(
                    f"event {event_id!r} has {kind} time but sample "
                    f"{sample.sample_id!r} cutoff is {sample_kind}"
                )
            try:
                in_window = sample.window_start <= time_value < sample.cutoff
                bool(in_window)
            except TypeError as exc:
                raise EventValidationError(
                    f"event time {time_value!r} cannot be compared to the window of "
                    f"sample {sample.sample_id!r}"
                ) from exc

    # ------------------------------------------------------------------ #
    # Window construction and signatures
    # ------------------------------------------------------------------ #
    def _window_rows(self, sample):
        """Return the live events inside ``sample``'s [start, cutoff) window."""
        rows = []
        for event_id, record in self._authority.items():
            if not record["alive"] or record["entity"] != sample.entity:
                continue
            if sample.window_start <= record["time"] < sample.cutoff:
                rows.append((event_id, record))
        rows.sort(key=lambda item: (item[1]["time"], _tagged(item[0])))
        return rows

    @staticmethod
    def _content_signature(rows):
        """
        Hash the actual window input (time, kind, value of every live event
        in canonical order).  Event identities are deliberately not part of
        the content hash: two windows with identical input series must be
        treated as unchanged even if the contributing event ids differ.
        """
        digest = hashlib.sha256()
        for _, record in rows:
            digest.update(_canonical_time(record["time"]).encode("utf-8"))
            digest.update(b"|")
            digest.update(str(record["kind"]).encode("utf-8"))
            digest.update(b"|")
            digest.update(repr(float(record["value"])).encode("utf-8"))
            digest.update(b"\n")
        return digest.hexdigest()

    def _empty_source(self, sample, last_extracted_generation=None):
        return WindowSource(
            sample_id=sample.sample_id,
            entity=sample.entity,
            window_start=sample.window_start,
            window_end_exclusive=sample.cutoff,
            event_ids=(),
            content_signature=self._content_signature([]),
            last_extracted_generation=last_extracted_generation,
        )

    def _source_for(self, sample, rows, last_extracted_generation):
        return WindowSource(
            sample_id=sample.sample_id,
            entity=sample.entity,
            window_start=sample.window_start,
            window_end_exclusive=sample.cutoff,
            event_ids=tuple(event_id for event_id, _ in rows),
            content_signature=self._content_signature(rows),
            last_extracted_generation=last_extracted_generation,
        )

    def _samples_covering(self, entity, time_value):
        """Sample ids whose window contains the (entity, time) point."""
        result = []
        for sample in self._entity_samples.get(entity, ()):
            if sample.window_start <= time_value < sample.cutoff:
                result.append(sample.sample_id)
        return result

    # ------------------------------------------------------------------ #
    # Native extraction (never hand-computed)
    # ------------------------------------------------------------------ #
    def _native_extract(self, frame):
        """Run the native tsfresh extractor on a long-format frame."""
        return extract_features(
            frame,
            default_fc_parameters=self._config["default_fc_parameters"],
            kind_to_fc_parameters=self._config["kind_to_fc_parameters"],
            column_id=_S_ID,
            column_sort=_S_SORT,
            column_kind=_S_KIND,
            column_value=_S_VALUE,
            n_jobs=1,
            disable_progressbar=True,
            show_warnings=False,
            impute_function=None,
        )

    def _register_columns(self, native_result):
        """
        Adopt any columns the native extractor produced that we do not know
        yet.  New columns are appended in the *native* output order, which
        keeps incremental matrices comparable with a full re-extraction
        (native column order follows the fc-parameters dictionary, not
        alphabetical order).
        """
        new_columns = [c for c in native_result.columns if c not in self._declared_columns]
        if new_columns:
            self._declared_columns.extend(new_columns)
            self._features = self._features.reindex(
                columns=self._declared_columns
            )

    def _probe_kind(self, kind):
        """
        Discover the native feature columns produced for ``kind`` by running
        the real extractor on a tiny synthetic series.  This keeps the
        declared columns stable even for samples whose windows are empty or
        never contain that kind.
        """
        if kind in self._probed_kinds:
            return
        probe = pd.DataFrame(
            {
                _S_ID: ["__probe__", "__probe__"],
                _S_SORT: [0, 1],
                _S_KIND: [kind, kind],
                _S_VALUE: [1.0, 2.0],
            }
        )
        result = self._native_extract(probe)
        self._probed_kinds.add(kind)
        self._register_columns(result)

    def _probe_configured_kinds(self):
        kind_config = self._config["kind_to_fc_parameters"]
        if kind_config:
            for kind in kind_config:
                self._probe_kind(str(kind))

    def _probe_event_kinds(self, records):
        # Probe first-seen kinds in a stable, sorted order.  The native
        # extractor groups columns by kind (first-seen order) and, within a
        # kind, by fc-parameters dictionary order; probing each new kind up
        # front makes the declared columns deterministic regardless of the
        # order in which events happen to arrive.  Delete records carry no
        # location and are excluded.
        seen = sorted({record[KIND] for record in records if record[OP] == _OP_UPSERT})
        for kind in seen:
            if kind not in self._probed_kinds:
                self._probe_kind(kind)

    def _extract_samples(self, sample_ids):
        """
        Re-extract the given samples natively.  Samples with empty windows
        are never sent to the extractor (it has no rows to work with); their
        rows are explicitly filled with NaN while keeping every declared
        column.
        """
        frames = []
        empty_ids = []
        window_rows = {}
        for sample in self._samples:
            if sample.sample_id not in sample_ids:
                continue
            rows = self._window_rows(sample)
            window_rows[sample.sample_id] = rows
            if not rows:
                empty_ids.append(sample.sample_id)
                continue
            frame = pd.DataFrame(
                {
                    _S_ID: [sample.sample_id] * len(rows),
                    _S_SORT: [record["time"] for _, record in rows],
                    _S_KIND: [record["kind"] for _, record in rows],
                    _S_VALUE: [record["value"] for _, record in rows],
                }
            )
            frames.append(frame)

        if frames:
            batch = pd.concat(frames, ignore_index=True)
            native_result = self._native_extract(batch)
            self._register_columns(native_result)
            for sample_id in native_result.index:
                self._features.loc[sample_id] = native_result.loc[
                    sample_id
                ].reindex(self._declared_columns)

        # Empty windows: declared columns retained, explicitly NaN.
        if empty_ids:
            self._features.loc[empty_ids, :] = np.nan

        return window_rows

    # ------------------------------------------------------------------ #
    # The incremental update
    # ------------------------------------------------------------------ #
    def apply_events(self, events, commit=None):
        """
        Apply a batch of new, corrected or deleted events and incrementally
        refresh the feature table.

        :param events: DataFrame with the columns ``event_id``,
            ``revision``, ``op``, ``entity``, ``time``, ``kind``,
            ``value``.  ``op`` is one of ``upsert``/``insert``/``update``
            (aliases) or ``delete``/``remove``.  Delete rows only need
            ``event_id`` and ``revision``; the location columns are ignored.
        :param commit: if ``True``, persist the new generation to the store
            (requires ``store_dir`` at construction).  Defaults to ``True``
            when a store is configured and ``False`` otherwise.
        :return: :class:`UpdateReport` describing what changed.
        :raises EventValidationError: malformed batch or sample declaration.
        :raises RevisionConflictError: a revision number is reused with a
            different payload.  The table state is left completely
            untouched in that case.
        """
        if commit is None:
            commit = self._store is not None
        if commit and self._store is None:
            raise IncrementalError(
                "commit=True requires store_dir to be configured at construction"
            )

        normalized = self._validate_batch(events)

        # Snapshot everything that may change so the call is all-or-nothing.
        snapshot = self._snapshot()
        try:
            report = self._apply_normalized(normalized, commit)
        except BaseException:
            self._restore(snapshot)
            raise
        return report

    def _snapshot(self):
        return {
            "authority": {
                eid: dict(record) for eid, record in self._authority.items()
            },
            "features": self._features.copy(),
            "provenance": dict(self._provenance),
            "declared_columns": list(self._declared_columns),
            "probed_kinds": set(self._probed_kinds),
            "generation": self._generation,
            "time_kind": self._time_kind,
        }

    def _restore(self, snapshot):
        self._authority = {
            eid: dict(record) for eid, record in snapshot["authority"].items()
        }
        self._features = snapshot["features"].copy()
        self._provenance = dict(snapshot["provenance"])
        self._declared_columns = list(snapshot["declared_columns"])
        self._probed_kinds = set(snapshot["probed_kinds"])
        self._generation = snapshot["generation"]
        self._time_kind = snapshot["time_kind"]

    def _apply_normalized(self, normalized, commit):
        received = len(normalized)
        effective = 0
        stale_skipped = 0
        duplicate_skipped = 0
        migrated = []
        candidate_points = []  # (entity, time) old/new locations to recheck

        for record in normalized:
            eid = record[EVENT_ID]
            revision = record[REVISION]
            op = record[OP]
            existing = self._authority.get(eid)

            if op == _OP_UPSERT:
                decision = self._stage_upsert(eid, revision, record, existing)
            else:
                decision = self._stage_delete(eid, revision, existing)

            kind, old_point, new_point, migration = decision
            if kind == "stale":
                stale_skipped += 1
                continue
            if kind == "duplicate":
                duplicate_skipped += 1
                continue

            effective += 1
            if migration:
                migrated.append(eid)
            if old_point is not None:
                candidate_points.append(old_point)
            if new_point is not None:
                candidate_points.append(new_point)

        if effective == 0:
            # No authority change: no new generation, no recomputation.
            return UpdateReport(
                generation=self._generation,
                received=received,
                effective=0,
                stale_skipped=stale_skipped,
                duplicate_skipped=duplicate_skipped,
                affected_samples=(),
                recomputed_samples=(),
                unchanged_samples=(),
                migrated_events=tuple(migrated),
                committed=False,
            )

        # Discover any new kind's declared columns via the native extractor
        # before rebuilding windows.
        self._probe_event_kinds(normalized)

        affected = set()
        for entity, time_value in candidate_points:
            affected.update(self._samples_covering(entity, time_value))

        recomputed_ids = []
        unchanged_ids = []
        new_generation = self._generation + 1

        for sample in self._samples:
            if sample.sample_id not in affected:
                continue
            rows = self._window_rows(sample)
            new_signature = self._content_signature(rows)
            old_source = self._provenance.get(sample.sample_id)
            if (
                old_source is not None
                and old_source.content_signature == new_signature
            ):
                # Actual window input unchanged: keep the old feature row.
                # Provenance membership is still refreshed.
                unchanged_ids.append(sample.sample_id)
                self._provenance[sample.sample_id] = self._source_for(
                    sample, rows, old_source.last_extracted_generation
                )
            else:
                recomputed_ids.append(sample.sample_id)

        if recomputed_ids:
            window_rows = self._extract_samples(set(recomputed_ids))
            for sample in self._samples:
                if sample.sample_id not in recomputed_ids:
                    continue
                self._provenance[sample.sample_id] = self._source_for(
                    sample,
                    window_rows[sample.sample_id],
                    last_extracted_generation=new_generation,
                )

        self._features = self._features.reindex(columns=self._declared_columns)
        self._generation = new_generation
        committed = self._commit(recomputed_ids, tuple(sorted(affected, key=repr)))

        return UpdateReport(
            generation=self._generation,
            received=received,
            effective=effective,
            stale_skipped=stale_skipped,
            duplicate_skipped=duplicate_skipped,
            affected_samples=tuple(
                s.sample_id for s in self._samples if s.sample_id in affected
            ),
            recomputed_samples=tuple(recomputed_ids),
            unchanged_samples=tuple(unchanged_ids),
            migrated_events=tuple(migrated),
            committed=committed,
        )

    def _stage_upsert(self, event_id, revision, record, existing):
        payload = {
            ENTITY: record[ENTITY],
            TIME: record[TIME],
            KIND: record[KIND],
            VALUE: record[VALUE],
        }
        if existing is None:
            self._authority[event_id] = {
                **payload,
                REVISION: revision,
                "alive": True,
            }
            return "effective", None, (payload[ENTITY], payload[TIME]), False

        if revision < existing[REVISION]:
            return "stale", None, None, False

        if revision == existing[REVISION]:
            same_payload = (
                existing[ENTITY] == payload[ENTITY]
                and existing[TIME] == payload[TIME]
                and existing[KIND] == payload[KIND]
                and existing[VALUE] == payload[VALUE]
            )
            if existing["alive"] and same_payload:
                return "duplicate", None, None, False
            # Same revision number, different claim -> hard conflict.
            raise RevisionConflictError(
                f"event {event_id!r} revision {revision} was already recorded with a "
                "different payload or state; reusing a revision number with another "
                "value is not allowed"
            )

        # Newer revision wins.
        old_point = (
            (existing[ENTITY], existing[TIME]) if existing["alive"] else None
        )
        # A migration is any change of the event's location (entity or
        # time); a value/kind correction at the same location is not one.
        migration = existing["alive"] and (
            existing[ENTITY] != payload[ENTITY]
            or existing[TIME] != payload[TIME]
        )
        self._authority[event_id] = {
            **payload,
            REVISION: revision,
            "alive": True,
        }
        return "effective", old_point, (payload[ENTITY], payload[TIME]), migration

    def _stage_delete(self, event_id, revision, existing):
        if existing is None:
            # Record a tombstone so that any later lower revision is stale.
            self._authority[event_id] = {
                ENTITY: None,
                TIME: None,
                KIND: None,
                VALUE: None,
                REVISION: revision,
                "alive": False,
            }
            return "effective", None, None, False

        if revision < existing[REVISION]:
            return "stale", None, None, False

        if revision == existing[REVISION]:
            if not existing["alive"]:
                return "duplicate", None, None, False
            raise RevisionConflictError(
                f"event {event_id!r} revision {revision} is already recorded as an "
                "upsert; a delete cannot reuse the same revision number"
            )

        old_point = (
            (existing[ENTITY], existing[TIME]) if existing["alive"] else None
        )
        self._authority[event_id] = {
            ENTITY: None,
            TIME: None,
            KIND: None,
            VALUE: None,
            REVISION: revision,
            "alive": False,
        }
        return "effective", old_point, None, False

    # ------------------------------------------------------------------ #
    # Query / export
    # ------------------------------------------------------------------ #
    @property
    def generation(self):
        """Current committed generation number."""
        return self._generation

    @property
    def features(self):
        """The real native feature matrix (one row per declared sample)."""
        return self._features.copy()

    @property
    def declared_feature_columns(self):
        """All feature columns the table ever declared (stable schema)."""
        return list(self._declared_columns)

    @property
    def samples(self):
        return tuple(self._samples)

    @property
    def store(self):
        return self._store

    def get_feature(self, sample_id, feature_name):
        """Return one feature value, or ``NaN`` for an empty/missing cell."""
        if sample_id not in self._features.index:
            raise KeyError(f"unknown sample_id {sample_id!r}")
        if feature_name not in self._features.columns:
            raise KeyError(f"unknown feature column {feature_name!r}")
        value = self._features.loc[sample_id, feature_name]
        return float(value) if not pd.isna(value) else np.nan

    def window_source(self, sample_id):
        """Return the :class:`WindowSource` provenance for one sample."""
        if sample_id not in self._provenance:
            raise KeyError(f"unknown sample_id {sample_id!r}")
        return self._provenance[sample_id]

    def window_sources(self):
        """Return all window provenance records as a DataFrame."""
        rows = []
        for sample in self._samples:
            source = self._provenance[sample.sample_id]
            rows.append(
                {
                    SAMPLE_ID: source.sample_id,
                    ENTITY: source.entity,
                    "window_start": source.window_start,
                    "window_end_exclusive": source.window_end_exclusive,
                    "event_ids": list(source.event_ids),
                    "n_events": len(source.event_ids),
                    "content_signature": source.content_signature,
                    "last_extracted_generation": source.last_extracted_generation,
                }
            )
        return pd.DataFrame(rows)

    def window_events(self, sample_id):
        """
        Return the live events currently making up the window of
        ``sample_id`` as a DataFrame in the canonical event schema, ordered
        by time then event id.
        """
        sample = next((s for s in self._samples if s.sample_id == sample_id), None)
        if sample is None:
            raise KeyError(f"unknown sample_id {sample_id!r}")
        rows = self._window_rows(sample)
        return pd.DataFrame(
            [
                {
                    EVENT_ID: event_id,
                    ENTITY: record[ENTITY],
                    TIME: record[TIME],
                    KIND: record[KIND],
                    VALUE: record[VALUE],
                    REVISION: record[REVISION],
                }
                for event_id, record in rows
            ],
            columns=[EVENT_ID, ENTITY, TIME, KIND, VALUE, REVISION],
        )

    def authority_events(self):
        """Return the whole raw authority (including tombstones) as a frame."""
        rows = []
        for event_id, record in self._authority.items():
            rows.append(
                {
                    EVENT_ID: event_id,
                    REVISION: record[REVISION],
                    OP: _OP_DELETE if not record["alive"] else _OP_UPSERT,
                    ENTITY: record[ENTITY],
                    TIME: record[TIME],
                    KIND: record[KIND],
                    VALUE: record[VALUE],
                }
            )
        return pd.DataFrame(
            rows, columns=[EVENT_ID, REVISION, OP, ENTITY, TIME, KIND, VALUE]
        )

    def export(self, export_dir):
        """
        Export the current generation to ``export_dir`` as CSV files plus a
        cloudpickled native configuration:

            * ``features.csv`` -- the real native feature matrix,
            * ``window_sources.csv`` -- the window provenance,
            * ``authority_events.csv`` -- the raw authoritative event log,
            * ``samples.csv`` -- the fixed sample declarations,
            * ``feature_config.pkl`` -- the native fc configuration.
        """
        import os

        import cloudpickle

        os.makedirs(export_dir, exist_ok=True)
        self._features.to_csv(os.path.join(export_dir, "features.csv"))
        self.window_sources().to_csv(
            os.path.join(export_dir, "window_sources.csv"), index=False
        )
        self.authority_events().to_csv(
            os.path.join(export_dir, "authority_events.csv"), index=False
        )
        pd.DataFrame(
            [
                {
                    SAMPLE_ID: s.sample_id,
                    ENTITY: s.entity,
                    CUTOFF: s.cutoff,
                    LOOKBACK: s.lookback_length,
                    "window_start": s.window_start,
                }
                for s in self._samples
            ]
        ).to_csv(os.path.join(export_dir, "samples.csv"), index=False)
        with open(os.path.join(export_dir, "feature_config.pkl"), "wb") as fh:
            fh.write(cloudpickle.dumps(self._config))
        return export_dir
