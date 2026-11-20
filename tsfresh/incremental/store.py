# -*- coding: utf-8 -*-
"""
Atomic, whole-generation persistence for the incremental rolling feature
table.

A *generation* is one immutable snapshot of everything needed to reconstruct
the table after a process restart:

    * the raw authoritative event log (the source of truth),
    * the native feature configuration,
    * the declared samples,
    * the extracted feature matrix and the per-sample window provenance.

Each generation is written into its own directory.  The generation becomes
visible only after all of its files have been durably written: a
``CURRENT`` pointer file is atomically replaced (via a sibling temporary
file and ``os.replace``) and fsynced.  A failure while writing therefore
leaves the previously committed generation completely intact -- there is
never a half-visible generation.  Reopening the directory always reads the
generation that ``CURRENT`` points at.
"""

import json
import os
import shutil

import cloudpickle

# File names inside a generation directory.
META_FILE = "meta.json"
AUTHORITY_FILE = "authority.pkl"
FEATURES_FILE = "features.pkl"
PROVENANCE_FILE = "provenance.pkl"
SAMPLES_FILE = "samples.pkl"
CONFIG_FILE = "config.pkl"

CURRENT_POINTER = "CURRENT"
TEMP_SUFFIX = ".tmp"
GEN_PREFIX = "gen-"


class StoreError(RuntimeError):
    """Raised when a generation cannot be written or read safely."""


def _atomic_write_bytes(path, data):
    """Write ``data`` to ``path`` atomically (temp sibling + os.replace)."""
    tmp_path = path + TEMP_SUFFIX
    with open(tmp_path, "wb") as fh:
        fh.write(data)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp_path, path)


def _dump_pickle(obj):
    return cloudpickle.dumps(obj, protocol=cloudpickle.DEFAULT_PROTOCOL)


class GenerationStore:
    """
    Filesystem-backed, append-only storage for whole table generations.

    The on-disk layout of ``store_dir`` is::

        store_dir/
            CURRENT                 # text file holding the current gen name
            gen-000000/
                meta.json           # human-readable generation metadata
                authority.pkl       # raw authoritative event log
                config.pkl          # native feature configuration
                samples.pkl         # declared, immutable sample definitions
                features.pkl        # real native feature matrix
                provenance.pkl      # window sources / content signatures
            gen-000001/
                ...

    The class is intentionally agnostic of the payload semantics: callers
    hand it (authority, config, samples, features, provenance) dicts and
    receive them back unchanged.  All payloads are serialized with
    cloudpickle so that native feature configurations containing callables
    (e.g. custom calculators) round-trip correctly.
    """

    def __init__(self, store_dir):
        self.store_dir = os.fspath(store_dir)
        os.makedirs(self.store_dir, exist_ok=True)

    # ------------------------------------------------------------------ #
    # Writing
    # ------------------------------------------------------------------ #
    def commit_generation(
        self,
        generation,
        authority,
        config,
        samples,
        features,
        provenance,
        extra_meta=None,
        keep_last=None,
    ):
        """
        Persist a complete generation atomically.

        Either the new generation is fully visible once this method returns
        or the previous generation stays in place unchanged.  The in-memory
        state of the caller is never modified here.

        :param generation: integer generation number (``>= 0``); the
            directory name is derived from it.
        :param authority: raw authoritative event log payload.
        :param config: native feature configuration payload.
        :param samples: declared sample payload.
        :param features: feature matrix payload.
        :param provenance: window provenance payload.
        :param extra_meta: optional JSON-serializable dict merged into
            ``meta.json`` (must contain only JSON-safe values).
        :param keep_last: if not ``None``, keep at most this many
            generations on disk and prune the older ones after the new
            generation has been committed.  ``None`` keeps every generation.
        """
        if not isinstance(generation, int) or generation < 0:
            raise StoreError(
                f"generation must be a non-negative integer, got {generation!r}"
            )
        if keep_last is not None:
            self._validate_keep_last(keep_last)

        gen_name = f"{GEN_PREFIX}{generation:06d}"
        gen_dir = os.path.join(self.store_dir, gen_name)
        tmp_dir = os.path.join(self.store_dir, gen_name + TEMP_SUFFIX)

        # A leftover temp directory from an earlier crashed attempt must not
        # be reused: build the new generation from scratch.
        if os.path.exists(tmp_dir):
            shutil.rmtree(tmp_dir)
        if os.path.exists(gen_dir):
            # Re-committing the same generation number is treated as an
            # error: generations are immutable and append-only.
            raise StoreError(f"generation {generation} already exists at {gen_dir}")

        os.makedirs(tmp_dir)
        try:
            payloads = {
                AUTHORITY_FILE: authority,
                CONFIG_FILE: config,
                FEATURES_FILE: features,
                PROVENANCE_FILE: provenance,
                SAMPLES_FILE: samples,
            }
            for file_name, payload in payloads.items():
                _atomic_write_bytes(
                    os.path.join(tmp_dir, file_name), _dump_pickle(payload)
                )

            meta = {"generation": generation}
            if extra_meta:
                # Fail loudly rather than silently dropping non-serializable
                # metadata before the generation becomes visible.
                json.dumps(extra_meta)
                meta.update(extra_meta)
            _atomic_write_bytes(
                os.path.join(tmp_dir, META_FILE),
                (json.dumps(meta, indent=2, sort_keys=True) + "\n").encode("utf-8"),
            )

            # Make every file durable before publishing the directory.
            dir_fd = os.open(tmp_dir, os.O_RDONLY)
            try:
                os.fsync(dir_fd)
            finally:
                os.close(dir_fd)

            os.replace(tmp_dir, gen_dir)
        except BaseException:
            # Roll back the incomplete generation, leaving the old one alone.
            if os.path.exists(tmp_dir):
                shutil.rmtree(tmp_dir, ignore_errors=True)
            raise

        # Atomically repoint CURRENT at the new generation.
        self._set_current(gen_name)

        if keep_last is not None:
            self.prune(keep_last=keep_last, protect_generation=generation)

        return gen_dir

    def _set_current(self, gen_name):
        pointer_path = os.path.join(self.store_dir, CURRENT_POINTER)
        _atomic_write_bytes(pointer_path, gen_name.encode("utf-8"))
        root_fd = os.open(self.store_dir, os.O_RDONLY)
        try:
            os.fsync(root_fd)
        finally:
            os.close(root_fd)

    # ------------------------------------------------------------------ #
    # Reading
    # ------------------------------------------------------------------ #
    def current_generation_name(self):
        """Return the directory name of the committed generation, or None."""
        pointer_path = os.path.join(self.store_dir, CURRENT_POINTER)
        if not os.path.exists(pointer_path):
            return None
        with open(pointer_path, "r", encoding="utf-8") as fh:
            name = fh.read().strip()
        if not name:
            return None
        if not os.path.isdir(os.path.join(self.store_dir, name)):
            raise StoreError(
                f"{CURRENT_POINTER} points at missing generation directory {name!r}"
            )
        return name

    def load_current(self):
        """
        Load and return the current generation as a dict with keys
        ``generation``, ``authority``, ``config``, ``samples``,
        ``features``, ``provenance`` and ``meta``.

        Raises :class:`StoreError` if no generation has ever been committed
        or the pointer is inconsistent.
        """
        gen_name = self.current_generation_name()
        if gen_name is None:
            raise StoreError(
                f"no committed generation found in {self.store_dir!r}"
            )
        return self.load_generation(gen_name)

    def load_generation(self, gen_name):
        gen_dir = os.path.join(self.store_dir, gen_name)
        if not os.path.isdir(gen_dir):
            raise StoreError(f"generation directory missing: {gen_dir!r}")

        def _read(file_name):
            with open(os.path.join(gen_dir, file_name), "rb") as fh:
                return cloudpickle.loads(fh.read())

        with open(os.path.join(gen_dir, META_FILE), "r", encoding="utf-8") as fh:
            meta = json.load(fh)

        return {
            "generation": meta["generation"],
            "authority": _read(AUTHORITY_FILE),
            "config": _read(CONFIG_FILE),
            "samples": _read(SAMPLES_FILE),
            "features": _read(FEATURES_FILE),
            "provenance": _read(PROVENANCE_FILE),
            "meta": meta,
        }

    def list_generations(self):
        """Return committed generation numbers in ascending order."""
        result = []
        for name in os.listdir(self.store_dir):
            if name.startswith(GEN_PREFIX) and os.path.isdir(
                os.path.join(self.store_dir, name)
            ):
                try:
                    result.append(int(name[len(GEN_PREFIX):]))
                except ValueError:
                    continue
        return sorted(result)

    # ------------------------------------------------------------------ #
    # Maintenance
    # ------------------------------------------------------------------ #
    @staticmethod
    def _validate_keep_last(keep_last):
        if not isinstance(keep_last, int) or keep_last < 1:
            raise StoreError(
                f"keep_last must be a positive integer or None, got {keep_last!r}"
            )

    def prune(self, keep_last, protect_generation=None):
        """
        Remove all but the newest ``keep_last`` generations.  The generation
        ``protect_generation`` (typically the one just committed) is never
        removed even if it would otherwise fall outside the window.
        """
        self._validate_keep_last(keep_last)
        generations = self.list_generations()
        if len(generations) <= keep_last:
            return
        to_remove = generations[: len(generations) - keep_last]
        for gen in to_remove:
            if gen == protect_generation:
                continue
            gen_dir = os.path.join(self.store_dir, f"{GEN_PREFIX}{gen:06d}")
            shutil.rmtree(gen_dir, ignore_errors=True)

    def cleanup_temp(self):
        """Remove leftover ``*.tmp`` generation directories from crashes."""
        for name in os.listdir(self.store_dir):
            path = os.path.join(self.store_dir, name)
            if name.endswith(TEMP_SUFFIX) and os.path.isdir(path):
                shutil.rmtree(path, ignore_errors=True)
