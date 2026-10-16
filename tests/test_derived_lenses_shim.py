"""Regression tests for the legacy derived_lenses compatibility module."""


def test_legacy_shim_exports_peak_lens_used_by_vision_harness():
    import derived_lenses
    from perception.lenses.peak_lens import PeakLens

    assert derived_lenses.PeakLens is PeakLens
