# -*- coding: utf-8 -*-
"""Direct unit tests for the atomic generation store."""

import os
import shutil
import tempfile

import cloudpickle
import pytest

from tsfresh.incremental import GenerationStore, StoreError


@pytest.fixture()
def store_dir():
    path = tempfile.mkdtemp(prefix="tsfresh_store_")
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)


def _payload(seed):
    return (
        {"records": {"e1": {"v": seed}}},
        {"default_fc_parameters": {"mean": None}, "kind_to_fc_parameters": None},
        {"samples": [{"sample_id": 1}]},
        {"features": [["x", seed]]},
        {1: {"content_signature": f"sig-{seed}"}},
    )


def test_commit_and_load_roundtrip(store_dir):
    store = GenerationStore(store_dir)
    assert store.current_generation_name() is None
    authority, config, samples, features, provenance = _payload(1)
    store.commit_generation(
        0, authority, config, samples, features, provenance,
        extra_meta={"note": "first"},
    )
    assert store.current_generation_name() == "gen-000000"
    loaded = store.load_current()
    assert loaded["generation"] == 0
    assert loaded["authority"] == authority
    assert loaded["config"] == config
    assert loaded["samples"] == samples
    assert loaded["features"] == features
    assert loaded["provenance"] == provenance
    assert loaded["meta"]["note"] == "first"


def test_load_without_generation_raises(store_dir):
    with pytest.raises(StoreError):
        GenerationStore(store_dir).load_current()


def test_committing_same_generation_twice_rejected(store_dir):
    store = GenerationStore(store_dir)
    store.commit_generation(0, *_payload(1))
    with pytest.raises(StoreError):
        store.commit_generation(0, *_payload(2))
    # the first generation is untouched
    assert store.load_current()["authority"]["records"]["e1"]["v"] == 1


def test_atomic_pointer_advances_and_old_generation_survives(store_dir):
    store = GenerationStore(store_dir)
    store.commit_generation(0, *_payload(1))
    store.commit_generation(1, *_payload(2))
    assert store.list_generations() == [0, 1]
    assert store.current_generation_name() == "gen-000001"
    # generation 0 is still readable and immutable on disk
    gen0 = store.load_generation("gen-000000")
    assert gen0["authority"]["records"]["e1"]["v"] == 1


def test_keep_last_pruning(store_dir):
    store = GenerationStore(store_dir)
    for gen in range(5):
        store.commit_generation(gen, *_payload(gen), keep_last=2)
    assert store.list_generations() == [3, 4]
    assert store.current_generation_name() == "gen-000004"


def test_cloudpickle_roundtrips_callable_config(store_dir):
    def custom(x):
        return float(len(x))

    config = {"default_fc_parameters": {custom: None}, "kind_to_fc_parameters": None}
    store = GenerationStore(store_dir)
    store.commit_generation(
        0,
        {"records": {}},
        config,
        {"samples": []},
        {"features": []},
        {},
    )
    loaded = store.load_current()
    restored_func = next(iter(loaded["config"]["default_fc_parameters"]))
    assert restored_func([1, 2, 3]) == 3.0


def test_bad_arguments(store_dir):
    store = GenerationStore(store_dir)
    with pytest.raises(StoreError):
        store.commit_generation(-1, *_payload(0))
    store.commit_generation(0, *_payload(0))
    with pytest.raises(StoreError):
        store.prune(keep_last=0)


def test_all_generation_files_present(store_dir):
    store = GenerationStore(store_dir)
    store.commit_generation(0, *_payload(0))
    gen_dir = os.path.join(store_dir, "gen-000000")
    assert sorted(os.listdir(gen_dir)) == [
        "authority.pkl",
        "config.pkl",
        "features.pkl",
        "meta.json",
        "provenance.pkl",
        "samples.pkl",
    ]
    # every pickle is independently readable
    for name in [
        "authority.pkl",
        "config.pkl",
        "features.pkl",
        "provenance.pkl",
        "samples.pkl",
    ]:
        with open(os.path.join(gen_dir, name), "rb") as fh:
            cloudpickle.loads(fh.read())


def test_dangling_pointer_detected(store_dir):
    store = GenerationStore(store_dir)
    store.commit_generation(0, *_payload(0))
    shutil.rmtree(os.path.join(store_dir, "gen-000000"))
    with pytest.raises(StoreError):
        store.current_generation_name()
