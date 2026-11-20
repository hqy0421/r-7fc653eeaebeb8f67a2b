"""
Incremental, rolling-window feature tables for tsfresh.

This subpackage adds an incremental feature store on top of the *native*
:func:`tsfresh.extract_features` extraction machinery.  It is meant for
event streams that are backfilled or corrected after a feature table has
already been built: instead of recomputing every sample, only the samples
whose rolling-window input actually changed are handed back to the native
extractor, while all unchanged rows are preserved.

Main entry point:

    from tsfresh.incremental import IncrementalRollingFeatureTable

The original tsfresh API (``tsfresh.extract_features`` etc.) is untouched.
"""

from tsfresh.incremental.rolling import (
    EventValidationError,
    IncrementalError,
    IncrementalRollingFeatureTable,
    RevisionConflictError,
    StoreError,
    UpdateReport,
    WindowSource,
)
from tsfresh.incremental.store import GenerationStore

__all__ = [
    "IncrementalRollingFeatureTable",
    "GenerationStore",
    "UpdateReport",
    "WindowSource",
    "IncrementalError",
    "EventValidationError",
    "RevisionConflictError",
    "StoreError",
]
