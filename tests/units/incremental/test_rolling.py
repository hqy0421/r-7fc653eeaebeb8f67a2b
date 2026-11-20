# -*- coding: utf-8 -*-
"""
Tests for the incremental rolling feature table
(``tsfresh.incremental``).
"""

import os
import shutil
import tempfile

import numpy as np
import pandas as pd
import pytest

from tsfresh import extract_features
from tsfresh.feature_extraction.settings import MinimalFCParameters
from tsfresh.incremental import (
    EventValidationError,
    GenerationStore,
    IncrementalRollingFeatureTable,
    RevisionConflictError,
    StoreError,
)


# ---------------------------------------------------------------------- #
# Fixtures / helpers
# ---------------------------------------------------------------------- #
SIMPLE_CONFIG = {"mean": None, "sum_values": None, "maximum": None, "minimum": None}


def make_samples():
    return pd.DataFrame(
        {
            "sample_id": [1, 2, 3],
            "entity": ["E", "E", "F"],
            "cutoff": [10.0, 20.0, 10.0],
            "lookback_length": [10.0, 10.0, 10.0],
        }
    )


def make_events():
    # E: e1(t=1,a,3), e2(t=5,a,7), e3(t=15,b,2); F: e4(t=2,a,4)
    return pd.DataFrame(
        {
            "event_id": ["e1", "e2", "e3", "e4"],
            "revision": [0, 0, 0, 0],
            "op": ["upsert"] * 4,
            "entity": ["E", "E", "E", "F"],
            "time": [1.0, 5.0, 15.0, 2.0],
            "kind": ["a", "a", "b", "a"],
            "value": [3.0, 7.0, 2.0, 4.0],
        }
    )


def native_reference(samples, events, config, kind_config=None):
    """Independent reference: rebuild every window and call native extraction."""
    frames = []
    for sample in samples.itertuples(index=False):
        live = events[
            (events["entity"] == sample.entity)
            & (events["time"] >= sample.cutoff - sample.lookback_length)
            & (events["time"] < sample.cutoff)
        ]
        for row in live.itertuples(index=False):
            frames.append(
                {
                    "sample_id": sample.sample_id,
                    "time": row.time,
                    "kind": row.kind,
                    "value": row.value,
                }
            )
    if not frames:
        return pd.DataFrame(
            index=pd.Index(
                list(samples["sample_id"]), name="sample_id"
            ),
            dtype=float,
        )
    frame = pd.DataFrame(frames)
    result = extract_features(
        frame,
        column_id="sample_id",
        column_sort="time",
        column_kind="kind",
        column_value="value",
        default_fc_parameters=config,
        kind_to_fc_parameters=kind_config,
        n_jobs=1,
        disable_progressbar=True,
        show_warnings=False,
        impute_function=None,
    )
    result.index.name = "sample_id"
    # The native extractor omits ids whose windows are empty; reindex back
    # onto the full declared sample set (NaN rows) to match the table,
    # which always keeps every declared sample row.
    full_index = pd.Index(list(samples["sample_id"]), name="sample_id")
    result = result.reindex(full_index)
    return result


@pytest.fixture()
def tmp_store():
    path = tempfile.mkdtemp(prefix="tsfresh_inc_")
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)


# ---------------------------------------------------------------------- #
# Construction / generation 0
# ---------------------------------------------------------------------- #
class TestConstruction:
    def test_empty_authority_generation_zero(self, tmp_store):
        table = IncrementalRollingFeatureTable(
            make_samples(),
            default_fc_parameters=SIMPLE_CONFIG,
            store_dir=os.path.join(tmp_store, "store"),
        )
        assert table.generation == 0
        # with no kind seen yet there are no declared columns, but every
        # declared sample already has a row and empty-window provenance
        assert table.features.shape[0] == 3
        assert table.features.columns.tolist() == []
        source = table.window_source(1)
        assert source.event_ids == ()
        assert source.window_end_exclusive == 10.0
        assert source.window_start == 0.0

    def test_empty_authority_with_declared_kinds_keeps_columns(self, tmp_store):
        table = IncrementalRollingFeatureTable(
            make_samples(),
            default_fc_parameters=SIMPLE_CONFIG,
            kind_to_fc_parameters={"a": SIMPLE_CONFIG},
            store_dir=os.path.join(tmp_store, "store"),
        )
        assert set(table.features.columns) == {
            "a__mean",
            "a__sum_values",
            "a__maximum",
            "a__minimum",
        }
        assert table.features.shape == (3, 4)
        assert table.features.isna().all().all()

    def test_initial_events_equivalent_to_native(self, tmp_store):
        table = IncrementalRollingFeatureTable(
            make_samples(),
            default_fc_parameters=SIMPLE_CONFIG,
            initial_events=make_events(),
            store_dir=os.path.join(tmp_store, "store"),
        )
        reference = native_reference(make_samples(), make_events(), SIMPLE_CONFIG)
        result = table.features
        pd.testing.assert_frame_equal(
            result.sort_index(), reference.sort_index(), check_dtype=False
        )

    def test_left_closed_right_open_boundary(self):
        samples = pd.DataFrame(
            {
                "sample_id": [1],
                "entity": ["E"],
                "cutoff": [10.0],
                "lookback_length": [5.0],
            }
        )
        events = pd.DataFrame(
            {
                "event_id": ["at_start", "at_end", "inside"],
                "revision": [0, 0, 0],
                "op": ["upsert"] * 3,
                "entity": ["E"] * 3,
                "time": [5.0, 10.0, 7.0],  # [5, 10): start included, end excluded
                "kind": ["a"] * 3,
                "value": [1.0, 99.0, 2.0],
            }
        )
        table = IncrementalRollingFeatureTable(
            samples, default_fc_parameters=SIMPLE_CONFIG, initial_events=events
        )
        # boundary event at cutoff must be excluded
        assert table.get_feature(1, "a__maximum") == 2.0
        assert set(table.window_source(1).event_ids) == {"at_start", "inside"}

    def test_missing_kind_keeps_declared_nan_columns(self):
        samples = make_samples().iloc[:1]
        table = IncrementalRollingFeatureTable(
            samples,
            default_fc_parameters=SIMPLE_CONFIG,
            kind_to_fc_parameters={"a": SIMPLE_CONFIG, "b": SIMPLE_CONFIG},
        )
        columns = set(table.features.columns)
        assert {"a__mean", "b__mean", "a__maximum", "b__maximum"} <= columns
        assert table.features.isna().all().all()

    def test_bad_sample_declaration(self):
        bad = pd.DataFrame(
            {"sample_id": [1, 1], "entity": ["E", "E"], "cutoff": [1.0, 2.0],
             "lookback_length": [1.0, 1.0]}
        )
        with pytest.raises(EventValidationError):
            IncrementalRollingFeatureTable(bad, default_fc_parameters=SIMPLE_CONFIG)

        missing = pd.DataFrame({"sample_id": [1]})
        with pytest.raises(EventValidationError):
            IncrementalRollingFeatureTable(missing, default_fc_parameters=SIMPLE_CONFIG)

    def test_datetime_windows(self):
        samples = pd.DataFrame(
            {
                "sample_id": ["d1"],
                "entity": ["E"],
                "cutoff": [pd.Timestamp("2024-01-10")],
                "lookback_length": [pd.Timedelta(days=7)],
            }
        )
        events = pd.DataFrame(
            {
                "event_id": ["x", "y"],
                "revision": [0, 0],
                "op": ["upsert", "upsert"],
                "entity": ["E", "E"],
                "time": [pd.Timestamp("2024-01-03"), pd.Timestamp("2024-01-10")],
                "kind": ["a", "a"],
                "value": [5.0, 50.0],
            }
        )
        table = IncrementalRollingFeatureTable(
            samples, default_fc_parameters=SIMPLE_CONFIG, initial_events=events
        )
        # 2024-01-03 is at the inclusive start; 2024-01-10 at exclusive end
        assert table.get_feature("d1", "a__sum_values") == 5.0
        assert table.window_source("d1").event_ids == ("x",)


# ---------------------------------------------------------------------- #
# Incremental updates
# ---------------------------------------------------------------------- #
class TestIncrementalUpdates:
    def test_only_changed_samples_are_recomputed(self):
        table = IncrementalRollingFeatureTable(
            make_samples(),
            default_fc_parameters=SIMPLE_CONFIG,
            initial_events=make_events(),
        )
        # event affecting only entity F -> sample 3
        new_event = pd.DataFrame(
            {
                "event_id": ["e5"],
                "revision": [0],
                "op": ["upsert"],
                "entity": ["F"],
                "time": [3.0],
                "kind": ["a"],
                "value": [9.0],
            }
        )
        report = table.apply_events(new_event)
        assert report.affected_samples == (3,)
        assert report.recomputed_samples == (3,)
        assert report.unchanged_samples == ()
        # untouched rows preserved exactly
        before = table.features

        report2 = table.apply_events(
            pd.DataFrame(
                {
                    "event_id": ["e6"],
                    "revision": [0],
                    "op": ["upsert"],
                    "entity": ["G"],  # entity no sample watches
                    "time": [3.0],
                    "kind": ["a"],
                    "value": [1.0],
                }
            )
        )
        assert report2.affected_samples == ()
        assert report2.recomputed_samples == ()
        pd.testing.assert_frame_equal(table.features, before)

    def test_migration_affects_old_and_new_windows(self):
        table = IncrementalRollingFeatureTable(
            make_samples(),
            default_fc_parameters=SIMPLE_CONFIG,
            initial_events=make_events(),
        )
        # e2 (t=5) moves to t=15: leaves sample 1 window, enters sample 2
        moved = pd.DataFrame(
            {
                "event_id": ["e2"],
                "revision": [1],
                "op": ["update"],
                "entity": ["E"],
                "time": [15.0],
                "kind": ["a"],
                "value": [7.0],
            }
        )
        report = table.apply_events(moved)
        assert set(report.affected_samples) == {1, 2}
        assert set(report.recomputed_samples) == {1, 2}
        assert report.migrated_events == ("e2",)
        assert "e2" not in table.window_source(1).event_ids
        assert "e2" in table.window_source(2).event_ids
        # sample 3 (entity F) untouched
        assert 3 not in report.affected_samples

    def test_cross_entity_migration(self):
        table = IncrementalRollingFeatureTable(
            make_samples(),
            default_fc_parameters=SIMPLE_CONFIG,
            initial_events=make_events(),
        )
        # e1 moves from E to F: sample 1 loses it, sample 3 gains it
        moved = pd.DataFrame(
            {
                "event_id": ["e1"],
                "revision": [1],
                "op": ["upsert"],
                "entity": ["F"],
                "time": [2.0],
                "kind": ["a"],
                "value": [3.0],
            }
        )
        report = table.apply_events(moved)
        assert set(report.affected_samples) == {1, 3}
        assert "e1" in table.window_source(3).event_ids
        assert "e1" not in table.window_source(1).event_ids

    def test_correction_same_window_unchanged_content_keeps_row(self):
        # two events with identical time/kind/value but different ids are
        # interchangeable as window *content*; replacing one must not
        # trigger native recomputation
        table = IncrementalRollingFeatureTable(
            make_samples(),
            default_fc_parameters=SIMPLE_CONFIG,
            initial_events=make_events(),
        )
        before = table.features.copy()
        # delete e2 (t=5,a,7) and insert e2b at the same time/kind/value
        batch = pd.DataFrame(
            {
                "event_id": ["e2", "e2b"],
                "revision": [1, 0],
                "op": ["delete", "upsert"],
                "entity": [None, "E"],
                "time": [None, 5.0],
                "kind": [None, "a"],
                "value": [None, 7.0],
            }
        )
        report = table.apply_events(batch)
        assert set(report.affected_samples) == {1}
        assert report.unchanged_samples == (1,)
        assert report.recomputed_samples == ()
        pd.testing.assert_frame_equal(table.features, before)
        # provenance still tracks the new membership
        assert "e2b" in table.window_source(1).event_ids
        assert "e2" not in table.window_source(1).event_ids

    def test_delete_event(self):
        table = IncrementalRollingFeatureTable(
            make_samples(),
            default_fc_parameters=SIMPLE_CONFIG,
            initial_events=make_events(),
        )
        report = table.apply_events(
            pd.DataFrame(
                {
                    "event_id": ["e1"],
                    "revision": [1],
                    "op": ["delete"],
                    "entity": [None],
                    "time": [None],
                    "kind": [None],
                    "value": [None],
                }
            )
        )
        assert report.effective == 1
        assert report.recomputed_samples == (1,)
        assert "e1" not in table.window_source(1).event_ids
        assert table.get_feature(1, "a__sum_values") == 7.0

    def test_deleting_all_events_yields_empty_window_with_columns(self):
        samples = make_samples().iloc[:1]
        table = IncrementalRollingFeatureTable(
            samples,
            default_fc_parameters=SIMPLE_CONFIG,
            initial_events=make_events(),
        )
        ids = ["e1", "e2"]
        batch = pd.DataFrame(
            {
                "event_id": ids,
                "revision": [1, 1],
                "op": ["delete"] * 2,
                "entity": [None] * 2,
                "time": [None] * 2,
                "kind": [None] * 2,
                "value": [None] * 2,
            }
        )
        table.apply_events(batch)
        row = table.features.loc[1]
        assert row.isna().all()
        # every column ever declared (for kinds a and b) is retained...
        assert len(row) == 8
        # ...including the kind-a columns that used to hold real values
        assert {
            "a__mean",
            "a__sum_values",
            "a__maximum",
            "a__minimum",
        } <= set(table.features.columns)
        assert table.window_source(1).event_ids == ()
        assert table.window_source(1).last_extracted_generation == table.generation

    def test_duplicate_and_stale_revisions_are_idempotent(self):
        table = IncrementalRollingFeatureTable(
            make_samples(),
            default_fc_parameters=SIMPLE_CONFIG,
            initial_events=make_events(),
        )
        gen_before = table.generation
        before = table.features.copy()
        # exact repeat of e1 revision 0
        report = table.apply_events(make_events().iloc[[0]])
        assert report.duplicate_skipped == 1
        assert report.effective == 0
        assert report.generation == gen_before
        pd.testing.assert_frame_equal(table.features, before)

        # advance e1 to revision 1 first
        table.apply_events(
            pd.DataFrame(
                {
                    "event_id": ["e1"],
                    "revision": [1],
                    "op": ["upsert"],
                    "entity": ["E"],
                    "time": [1.0],
                    "kind": ["a"],
                    "value": [4.0],
                }
            )
        )
        # older revision arriving late -> stale (revision 0 < current 1)
        stale = pd.DataFrame(
            {
                "event_id": ["e1"],
                "revision": [0],
                "op": ["upsert"],
                "entity": ["E"],
                "time": [1.0],
                "kind": ["a"],
                "value": [999.0],
            }
        )
        report = table.apply_events(stale)
        assert report.stale_skipped == 1
        assert report.effective == 0
        # the revision-1 value, not the stale one, is in effect
        assert table.get_feature(1, "a__sum_values") == 11.0

    def test_same_revision_different_value_conflicts(self):
        table = IncrementalRollingFeatureTable(
            make_samples(),
            default_fc_parameters=SIMPLE_CONFIG,
            initial_events=make_events(),
        )
        conflicting = pd.DataFrame(
            {
                "event_id": ["e1"],
                "revision": [0],  # revision 0 already exists with value 3
                "op": ["upsert"],
                "entity": ["E"],
                "time": [1.0],
                "kind": ["a"],
                "value": [999.0],
            }
        )
        with pytest.raises(RevisionConflictError):
            table.apply_events(conflicting)
        # state untouched
        assert table.generation == 0
        assert table.get_feature(1, "a__sum_values") == 10.0

    def test_delete_reusing_revision_conflicts(self):
        table = IncrementalRollingFeatureTable(
            make_samples().iloc[:1],
            default_fc_parameters=SIMPLE_CONFIG,
            initial_events=make_events(),
        )
        with pytest.raises(RevisionConflictError):
            table.apply_events(
                pd.DataFrame(
                    {
                        "event_id": ["e1"],
                        "revision": [0],
                        "op": ["delete"],
                        "entity": [None],
                        "time": [None],
                        "kind": [None],
                        "value": [None],
                    }
                )
            )

    def test_resurrection_with_newer_revision(self):
        table = IncrementalRollingFeatureTable(
            make_samples().iloc[:1],
            default_fc_parameters=SIMPLE_CONFIG,
            initial_events=make_events(),
        )
        table.apply_events(
            pd.DataFrame(
                {
                    "event_id": ["e1"],
                    "revision": [1],
                    "op": ["delete"],
                    "entity": [None],
                    "time": [None],
                    "kind": [None],
                    "value": [None],
                }
            )
        )
        assert table.get_feature(1, "a__sum_values") == 7.0
        # newer revision brings the event back
        table.apply_events(
            pd.DataFrame(
                {
                    "event_id": ["e1"],
                    "revision": [2],
                    "op": ["upsert"],
                    "entity": ["E"],
                    "time": [1.0],
                    "kind": ["a"],
                    "value": [3.0],
                }
            )
        )
        assert table.get_feature(1, "a__sum_values") == 10.0
        # the revision-1 delete is now stale
        report = table.apply_events(
            pd.DataFrame(
                {
                    "event_id": ["e1"],
                    "revision": [1],
                    "op": ["delete"],
                    "entity": [None],
                    "time": [None],
                    "kind": [None],
                    "value": [None],
                }
            )
        )
        assert report.stale_skipped == 1

    def test_many_incremental_batches_match_native_reference(self):
        table = IncrementalRollingFeatureTable(
            make_samples(), default_fc_parameters=SIMPLE_CONFIG
        )
        rng = np.random.RandomState(42)
        live = pd.DataFrame(columns=make_events().columns)

        batches = [
            make_events(),
            pd.DataFrame(
                {
                    "event_id": ["e5", "e6"],
                    "revision": [0, 0],
                    "op": ["upsert", "upsert"],
                    "entity": ["F", "E"],
                    "time": [8.0, 9.0],
                    "kind": ["b", "a"],
                    "value": [11.0, -4.0],
                }
            ),
            # correct e2
            pd.DataFrame(
                {
                    "event_id": ["e2"],
                    "revision": [1],
                    "op": ["update"],
                    "entity": ["E"],
                    "time": [6.0],
                    "kind": ["a"],
                    "value": [20.0],
                }
            ),
            # delete e3
            pd.DataFrame(
                {
                    "event_id": ["e3"],
                    "revision": [1],
                    "op": ["delete"],
                    "entity": [None],
                    "time": [None],
                    "kind": [None],
                    "value": [None],
                }
            ),
            # migrate e1 to F
            pd.DataFrame(
                {
                    "event_id": ["e1"],
                    "revision": [1],
                    "op": ["upsert"],
                    "entity": ["F"],
                    "time": [2.0],
                    "kind": ["a"],
                    "value": [3.0],
                }
            ),
        ]

        def apply_to_live(batch):
            kept = live[~live["event_id"].isin(batch["event_id"])]
            additions = batch[batch["op"] != "delete"][
                ["event_id", "revision", "op", "entity", "time", "kind", "value"]
            ]
            parts = [part for part in (kept, additions) if not part.empty]
            if not parts:
                return pd.DataFrame(
                    columns=[
                        "event_id",
                        "revision",
                        "op",
                        "entity",
                        "time",
                        "kind",
                        "value",
                    ]
                )
            return pd.concat(parts, ignore_index=True)

        for batch in batches:
            table.apply_events(batch)
            live = apply_to_live(batch)
            reference = native_reference(
                make_samples(), live, SIMPLE_CONFIG
            )
            pd.testing.assert_frame_equal(
                table.features.sort_index(),
                reference.sort_index(),
                check_dtype=False,
            )

    def test_validation_errors(self):
        table = IncrementalRollingFeatureTable(
            make_samples(), default_fc_parameters=SIMPLE_CONFIG
        )
        with pytest.raises(EventValidationError):
            table.apply_events(pd.DataFrame({"event_id": ["x"]}))
        with pytest.raises(EventValidationError):
            table.apply_events(
                pd.DataFrame(
                    {
                        "event_id": ["x"],
                        "revision": [-1],
                        "op": ["upsert"],
                        "entity": ["E"],
                        "time": [1.0],
                        "kind": ["a"],
                        "value": [1.0],
                    }
                )
            )
        with pytest.raises(EventValidationError):
            table.apply_events(
                pd.DataFrame(
                    {
                        "event_id": ["x"],
                        "revision": [0],
                        "op": ["nonsense"],
                        "entity": ["E"],
                        "time": [1.0],
                        "kind": ["a"],
                        "value": [1.0],
                    }
                )
            )
        with pytest.raises(EventValidationError):
            table.apply_events(
                pd.DataFrame(
                    {
                        "event_id": ["x"],
                        "revision": [0],
                        "op": ["upsert"],
                        "entity": ["E"],
                        "time": [1.0],
                        "kind": ["a"],
                        "value": [np.nan],
                    }
                )
            )
        # empty batch is rejected rather than silently bumping generations
        with pytest.raises(EventValidationError):
            table.apply_events(
                pd.DataFrame(
                    {
                        "event_id": pd.Series(dtype="object"),
                        "revision": pd.Series(dtype="int64"),
                        "op": pd.Series(dtype="object"),
                        "entity": pd.Series(dtype="object"),
                        "time": pd.Series(dtype="float64"),
                        "kind": pd.Series(dtype="object"),
                        "value": pd.Series(dtype="float64"),
                    }
                )
            )
        # numeric and datetime times cannot be mixed in one table
        with pytest.raises(EventValidationError):
            table.apply_events(
                pd.DataFrame(
                    {
                        "event_id": ["x"],
                        "revision": [0],
                        "op": ["upsert"],
                        "entity": ["E"],
                        "time": [pd.Timestamp("2024-01-01")],
                        "kind": ["a"],
                        "value": [1.0],
                    }
                )
            )

    def test_rollback_on_failure_leaves_state_intact(self):
        table = IncrementalRollingFeatureTable(
            make_samples(),
            default_fc_parameters=SIMPLE_CONFIG,
            initial_events=make_events(),
        )
        before = table.features.copy()
        gen_before = table.generation
        # a valid event followed by a conflicting one in the same batch
        bad_batch = pd.DataFrame(
            {
                "event_id": ["e9", "e1"],
                "revision": [0, 0],
                "op": ["upsert", "upsert"],
                "entity": ["F", "E"],
                "time": [1.0, 1.0],
                "kind": ["a", "a"],
                "value": [1.0, 999.0],  # e1 rev 0 conflicts
            }
        )
        with pytest.raises(RevisionConflictError):
            table.apply_events(bad_batch)
        assert table.generation == gen_before
        pd.testing.assert_frame_equal(table.features, before)
        assert "e9" not in table.authority_events()["event_id"].values


# ---------------------------------------------------------------------- #
# Native-only computation
# ---------------------------------------------------------------------- #
class TestNativeExtraction:
    def test_values_come_from_native_extractor(self):
        # Patch the native extractor and assert every feature goes through it.
        import tsfresh.incremental.rolling as rolling_mod

        table = IncrementalRollingFeatureTable(
            make_samples(), default_fc_parameters=SIMPLE_CONFIG
        )
        calls = []
        original = rolling_mod.extract_features

        def spy(*args, **kwargs):
            calls.append(kwargs["column_id"])
            return original(*args, **kwargs)

        rolling_mod.extract_features = spy
        try:
            table.apply_events(make_events())
        finally:
            rolling_mod.extract_features = original
        assert calls  # native extractor was really invoked
        # cross-check a concrete native value
        assert table.get_feature(1, "a__mean") == 5.0

    def test_native_minimal_parameters_work_end_to_end(self, tmp_store):
        table = IncrementalRollingFeatureTable(
            make_samples().iloc[:2],
            default_fc_parameters=MinimalFCParameters(),
            initial_events=make_events(),
            store_dir=os.path.join(tmp_store, "store"),
        )
        assert len(table.features.columns) > 10
        # missing-kind cells stay NaN, present-kind cells are real numbers
        assert table.features.isna().any().any()
        assert not table.features.isna().all().all()

    def test_kind_specific_configuration(self):
        table = IncrementalRollingFeatureTable(
            make_samples(),
            default_fc_parameters={},
            kind_to_fc_parameters={
                "a": {"mean": None},
                "b": {"sum_values": None},
            },
            initial_events=make_events(),
        )
        columns = set(table.features.columns)
        assert columns == {"a__mean", "b__sum_values"}
        reference = native_reference(
            make_samples(), make_events(), {},
            kind_config={"a": {"mean": None}, "b": {"sum_values": None}},
        )
        pd.testing.assert_frame_equal(
            table.features.sort_index(), reference.sort_index(), check_dtype=False
        )

    def test_new_kind_extends_declared_columns_and_backfills_nan(self):
        table = IncrementalRollingFeatureTable(
            make_samples().iloc[:2],
            default_fc_parameters=SIMPLE_CONFIG,
            initial_events=make_events(),
        )
        assert not any(c.startswith("c__") for c in table.features.columns)
        table.apply_events(
            pd.DataFrame(
                {
                    "event_id": ["e10"],
                    "revision": [0],
                    "op": ["upsert"],
                    "entity": ["E"],
                    "time": [2.0],
                    "kind": ["c"],
                    "value": [42.0],
                }
            )
        )
        assert any(c.startswith("c__") for c in table.features.columns)
        # sample 1's window contains the new kind-c event, sample 2's does not
        assert table.get_feature(1, "c__mean") == 42.0
        assert pd.isna(table.features.loc[2, "c__mean"])
        # previously existing columns are untouched
        assert table.get_feature(1, "a__mean") == 5.0


# ---------------------------------------------------------------------- #
# Persistence: generations, atomicity, reopen
# ---------------------------------------------------------------------- #
class TestPersistence:
    def test_generations_are_whole_and_reopen_is_consistent(self, tmp_store):
        store_dir = os.path.join(tmp_store, "store")
        table = IncrementalRollingFeatureTable(
            make_samples(),
            default_fc_parameters=SIMPLE_CONFIG,
            initial_events=make_events(),
            store_dir=store_dir,
        )
        table.apply_events(
            pd.DataFrame(
                {
                    "event_id": ["e5"],
                    "revision": [0],
                    "op": ["upsert"],
                    "entity": ["F"],
                    "time": [8.0],
                    "kind": ["a"],
                    "value": [6.0],
                }
            )
        )
        table.apply_events(
            pd.DataFrame(
                {
                    "event_id": ["e1"],
                    "revision": [1],
                    "op": ["delete"],
                    "entity": [None],
                    "time": [None],
                    "kind": [None],
                    "value": [None],
                }
            )
        )
        gen = table.generation
        features_before = table.features
        sources_before = table.window_sources()
        authority_before = table.authority_events()

        reopened = IncrementalRollingFeatureTable.load(store_dir)
        assert reopened.generation == gen
        pd.testing.assert_frame_equal(reopened.features, features_before)
        pd.testing.assert_frame_equal(
            reopened.window_sources().fillna(""),
            sources_before.fillna(""),
            check_dtype=False,
        )
        pd.testing.assert_frame_equal(
            reopened.authority_events().sort_values("event_id").reset_index(drop=True),
            authority_before.sort_values("event_id").reset_index(drop=True),
        )

        # continuing updates after reopen works identically
        reopened.apply_events(
            pd.DataFrame(
                {
                    "event_id": ["e7"],
                    "revision": [0],
                    "op": ["upsert"],
                    "entity": ["E"],
                    "time": [3.0],
                    "kind": ["a"],
                    "value": [-1.0],
                }
            )
        )
        assert reopened.generation == gen + 1
        assert reopened.get_feature(1, "a__minimum") == -1.0

        # and a second reopen sees exactly that state
        again = IncrementalRollingFeatureTable.load(store_dir)
        assert again.generation == gen + 1
        pd.testing.assert_frame_equal(again.features, reopened.features)
        assert again.get_feature(1, "a__minimum") == -1.0

        # queries and exports are identical before and after reopen
        export_before = os.path.join(tmp_store, "export_before")
        export_after = os.path.join(tmp_store, "export_after")
        reopened.export(export_before)
        again.export(export_after)
        for name in [
            "features.csv",
            "window_sources.csv",
            "authority_events.csv",
            "samples.csv",
        ]:
            with open(os.path.join(export_before, name)) as fh:
                before_text = fh.read()
            with open(os.path.join(export_after, name)) as fh:
                after_text = fh.read()
            assert before_text == after_text

    def test_store_contains_authority_config_and_features(self, tmp_store):
        store_dir = os.path.join(tmp_store, "store")
        table = IncrementalRollingFeatureTable(
            make_samples(),
            default_fc_parameters=SIMPLE_CONFIG,
            initial_events=make_events(),
            store_dir=store_dir,
        )
        payload = GenerationStore(store_dir).load_current()
        assert payload["generation"] == 0
        assert len(payload["authority"]["records"]) == 4
        assert payload["config"]["default_fc_parameters"] == SIMPLE_CONFIG
        assert len(payload["samples"]["samples"]) == 3
        pd.testing.assert_frame_equal(payload["features"], table.features)
        assert set(payload["provenance"].keys()) == {1, 2, 3}
        # all files exist
        gen_dir = os.path.join(store_dir, GenerationStore(store_dir).current_generation_name())
        for name in [
            "authority.pkl",
            "config.pkl",
            "features.pkl",
            "provenance.pkl",
            "samples.pkl",
            "meta.json",
        ]:
            assert os.path.exists(os.path.join(gen_dir, name))

    def test_failed_commit_keeps_old_generation(self, tmp_store):
        store_dir = os.path.join(tmp_store, "store")
        table = IncrementalRollingFeatureTable(
            make_samples(),
            default_fc_parameters=SIMPLE_CONFIG,
            initial_events=make_events(),
            store_dir=store_dir,
        )
        old_gen = table.generation
        old_features = table.features.copy()

        # Sabotage the commit: replace os.replace on the store instance.
        original_replace = os.replace

        def boom(src, dst):
            if "gen-" in str(dst):
                raise OSError("simulated disk failure")
            return original_replace(src, dst)

        os.replace = boom
        try:
            with pytest.raises(OSError):
                table.apply_events(
                    pd.DataFrame(
                        {
                            "event_id": ["e5"],
                            "revision": [0],
                            "op": ["upsert"],
                            "entity": ["F"],
                            "time": [8.0],
                            "kind": ["a"],
                            "value": [6.0],
                        }
                    )
                )
        finally:
            os.replace = original_replace

        # on-disk state unchanged
        reopened = IncrementalRollingFeatureTable.load(store_dir)
        assert reopened.generation == old_gen
        pd.testing.assert_frame_equal(reopened.features, old_features)

    def test_constructing_on_existing_store_requires_load(self, tmp_store):
        store_dir = os.path.join(tmp_store, "store")
        IncrementalRollingFeatureTable(
            make_samples(),
            default_fc_parameters=SIMPLE_CONFIG,
            store_dir=store_dir,
        )
        with pytest.raises(EventValidationError):
            IncrementalRollingFeatureTable(
                make_samples(),
                default_fc_parameters=SIMPLE_CONFIG,
                store_dir=store_dir,
            )

    def test_keep_last_prunes_generations(self, tmp_store):
        store_dir = os.path.join(tmp_store, "store")
        table = IncrementalRollingFeatureTable(
            make_samples(),
            default_fc_parameters=SIMPLE_CONFIG,
            initial_events=make_events(),
            store_dir=store_dir,
            keep_last=2,
        )
        for i in range(4):
            table.apply_events(
                pd.DataFrame(
                    {
                        "event_id": [f"e{i+10}"],
                        "revision": [0],
                        "op": ["upsert"],
                        "entity": ["F"],
                        "time": [1.0 + i],
                        "kind": ["a"],
                        "value": [float(i)],
                    }
                )
            )
        gens = GenerationStore(store_dir).list_generations()
        assert gens == [3, 4]
        reopened = IncrementalRollingFeatureTable.load(store_dir)
        assert reopened.generation == 4
        # pruning policy survives reopen: another update keeps only 2 gens
        reopened.apply_events(
            pd.DataFrame(
                {
                    "event_id": ["e20"],
                    "revision": [0],
                    "op": ["upsert"],
                    "entity": ["F"],
                    "time": [1.0],
                    "kind": ["a"],
                    "value": [8.0],
                }
            )
        )
        assert GenerationStore(store_dir).list_generations() == [4, 5]

    def test_export_roundtrip_files(self, tmp_store):
        store_dir = os.path.join(tmp_store, "store")
        export_dir = os.path.join(tmp_store, "export")
        table = IncrementalRollingFeatureTable(
            make_samples(),
            default_fc_parameters=SIMPLE_CONFIG,
            initial_events=make_events(),
            store_dir=store_dir,
        )
        table.export(export_dir)
        for name in [
            "features.csv",
            "window_sources.csv",
            "authority_events.csv",
            "samples.csv",
            "feature_config.pkl",
        ]:
            assert os.path.exists(os.path.join(export_dir, name))
        features = pd.read_csv(os.path.join(export_dir, "features.csv"), index_col=0)
        pd.testing.assert_frame_equal(features, table.features, check_dtype=False)

    def test_in_memory_table_does_not_persist(self):
        table = IncrementalRollingFeatureTable(
            make_samples(),
            default_fc_parameters=SIMPLE_CONFIG,
            initial_events=make_events(),
        )
        assert table.store is None
        report = table.apply_events(
            pd.DataFrame(
                {
                    "event_id": ["e5"],
                    "revision": [0],
                    "op": ["upsert"],
                    "entity": ["F"],
                    "time": [8.0],
                    "kind": ["a"],
                    "value": [6.0],
                }
            ),
            commit=False,
        )
        assert report.committed is False

    def test_commit_true_without_store_raises(self):
        from tsfresh.incremental import IncrementalError

        table = IncrementalRollingFeatureTable(
            make_samples(), default_fc_parameters=SIMPLE_CONFIG
        )
        with pytest.raises(IncrementalError):
            table.apply_events(make_events(), commit=True)


# ---------------------------------------------------------------------- #
# Query API
# ---------------------------------------------------------------------- #
class TestQueryAPI:
    def test_window_events_and_get_feature(self):
        table = IncrementalRollingFeatureTable(
            make_samples(),
            default_fc_parameters=SIMPLE_CONFIG,
            initial_events=make_events(),
        )
        window = table.window_events(1)
        assert list(window["event_id"]) == ["e1", "e2"]
        assert table.get_feature(1, "a__mean") == 5.0
        with pytest.raises(KeyError):
            table.get_feature(999, "a__mean")
        with pytest.raises(KeyError):
            table.get_feature(1, "nope__mean")

    def test_report_fields(self):
        table = IncrementalRollingFeatureTable(
            make_samples(), default_fc_parameters=SIMPLE_CONFIG
        )
        report = table.apply_events(make_events())
        assert report.generation == 1
        assert report.received == 4
        assert report.effective == 4
        assert report.committed is False
        assert len(report.affected_samples) == 3

    def test_repeated_queries_are_stable_copies(self):
        table = IncrementalRollingFeatureTable(
            make_samples(),
            default_fc_parameters=SIMPLE_CONFIG,
            initial_events=make_events(),
        )
        first = table.features
        first.iloc[0, 0] = -123456.0
        assert table.get_feature(1, "a__mean") == 5.0
