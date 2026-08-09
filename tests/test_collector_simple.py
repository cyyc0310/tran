"""Simple smoke test for collector module to verify basic functionality."""


def test_collector_import():
    """Collector module imports and exposes a callable."""
    from transcif.models.zeroshot.collector import collect_source_stacks

    assert callable(collect_source_stacks)


def test_collector_signature():
    """collect_source_stacks exposes the documented parameter list."""
    import inspect

    from transcif.models.zeroshot.collector import collect_source_stacks

    sig = inspect.signature(collect_source_stacks)
    params = list(sig.parameters.keys())
    expected = ["all_regions", "target_name", "seed", "device",
                "source_names", "progress"]
    assert params == expected, f"Expected {expected}, got {params}"


def test_collector_empty_sources():
    """Empty source list yields three empty lists."""
    from transcif.models.zeroshot.collector import collect_source_stacks

    cif_stacks, cif_true, names = collect_source_stacks(
        all_regions={},
        target_name="TARGET",
        seed=0,
        device=None,
        source_names=[],
        progress=False,
    )

    assert cif_stacks == []
    assert cif_true == []
    assert names == []
