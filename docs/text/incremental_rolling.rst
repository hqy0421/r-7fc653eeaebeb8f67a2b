.. _incremental-rolling-label:

=================================
Incremental rolling feature table
=================================

.. note::

    The incremental feature table lives in the :mod:`tsfresh.incremental`
    subpackage.  It is an additive layer: the ordinary tsfresh API
    (``tsfresh.extract_features``, the transformers, ...) is unchanged.

Motivation
==========

The classic tsfresh workflow computes a feature table from a snapshot of
the time series.  When the underlying events are *backfilled, corrected or
deleted* after the table has already been built, recomputing the whole
batch on every change is expensive, while keeping the old table means its
rows silently point at windows that no longer match the authoritative
event stream.

The :class:`~tsfresh.incremental.IncrementalRollingFeatureTable` solves
this by maintaining native tsfresh features over fixed rolling windows and
re-extracting only the rows whose actual window input changed.

Fixed samples, rolling windows
==============================

The samples are declared once and never change afterwards.  Each sample
fixes:

* a stable ``sample_id``,
* the ``entity`` whose events it watches,
* a ``cutoff`` time,
* a ``lookback_length``.

Its window is the half-open interval
``[cutoff - lookback_length, cutoff)``: the start is inclusive, the cutoff
is exclusive.  Both numeric and ``datetime`` cutoffs are supported (with a
numeric or ``pandas.Timedelta`` lookback respectively).

Events
======

Every event carries a stable ``event_id`` together with a monotonically
increasing integer ``revision``.  A batch is a ``pandas.DataFrame`` with the
columns

==================  ======================================================
column              meaning
==================  ======================================================
``event_id``        stable identity of the event
``revision``        monotonically increasing integer per ``event_id``
``op``              ``upsert`` (aliases: ``insert``, ``update``,
                    ``correct``) or ``delete`` (alias: ``remove``)
``entity``          entity the event belongs to
``time``            event timestamp (numeric or datetime-like)
``kind``            time series kind (as in the long input format)
``value``           numeric event value
==================  ======================================================

Delete rows only need ``event_id`` and ``revision``; their location columns
may be empty.

Revision handling
-----------------

* A revision **older** than the one already recorded for an event id is
  skipped as *stale* (reported in ``UpdateReport.stale_skipped``) and never
  reaches the native extractor.
* An **exact repeat** of the recorded revision is skipped as a *duplicate*
  (``UpdateReport.duplicate_skipped``).
* Reusing a known revision number with a **different payload or state**
  raises
  :class:`~tsfresh.incremental.RevisionConflictError`.  The whole batch is
  rejected and the table (and its on-disk generation) stays untouched.
* A newer revision always wins; a delete can be followed by a later
  upsert, which resurrects the event.

What gets recomputed
====================

For every effective event the table records both its old and its new
``(entity, time)`` location, so an event that migrates between windows
invalidates the old window *and* the new one.  Each candidate window is
then compared against its previous **content signature** -- a hash of the
time, kind and value of every live event it currently contains.

* If the signature is unchanged, the existing feature row is preserved
  byte-for-byte (reported as ``unchanged_samples``).  Window membership is
  still refreshed, so provenance stays accurate even when two events with
  identical content replace each other.
* Only windows whose real input changed are handed back to the native
  :func:`tsfresh.extract_features` (reported as ``recomputed_samples``).
  Features are never calculated by hand.
* Empty windows and kinds a sample never observed still keep every
  declared feature column; the corresponding cells are ``NaN``.

Generations and persistence
===========================

Every successful batch produces a new immutable *generation* containing,
whole:

* the raw authoritative event log (including tombstones),
* the native feature configuration,
* the fixed sample declarations,
* the real native feature matrix,
* the per-sample window provenance (``WindowSource``).

Generations are written atomically: all files of a new generation are
staged and fsynced before the ``CURRENT`` pointer is replaced, so a failed
commit leaves the previous generation completely authoritative.  Reopen
the directory with
:meth:`~tsfresh.incremental.IncrementalRollingFeatureTable.load`; queries,
CSV exports (``export``) and further updates then behave exactly as before
the restart.

Example
=======

.. code-block:: python

    import pandas as pd
    from tsfresh.incremental import IncrementalRollingFeatureTable

    samples = pd.DataFrame(
        {
            "sample_id": [1, 2],
            "entity": ["E", "E"],
            "cutoff": [10.0, 20.0],
            "lookback_length": [10.0, 10.0],
        }
    )

    events = pd.DataFrame(
        {
            "event_id": ["e1", "e2"],
            "revision": [0, 0],
            "op": ["upsert", "upsert"],
            "entity": ["E", "E"],
            "time": [2.0, 15.0],
            "kind": ["x", "x"],
            "value": [10.0, 30.0],
        }
    )

    table = IncrementalRollingFeatureTable(
        samples,
        default_fc_parameters={"mean": None, "sum_values": None},
        initial_events=events,
        store_dir="./feature_store",
    )

    # correct an event: only the windows whose content really changed
    # are re-extracted natively
    report = table.apply_events(
        pd.DataFrame(
            {
                "event_id": ["e1"],
                "revision": [1],
                "op": ["update"],
                "entity": ["E"],
                "time": [2.0],
                "kind": ["x"],
                "value": [40.0],
            }
        )
    )

    table.features          # the real native feature matrix
    table.window_sources()  # window boundaries, event ids and signatures
    table.export("./out")   # CSV export plus the native configuration

    # after a process restart
    table = IncrementalRollingFeatureTable.load("./feature_store")
