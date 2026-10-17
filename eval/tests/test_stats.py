import pytest
from eval.stats import wilson, clopper_pearson


def test_wilson_known_count():
    assert wilson(3, 100) == pytest.approx((0.010, 0.085), abs=0.0005)


def test_exact_known_count():
    assert clopper_pearson(3, 100) == pytest.approx((0.006, 0.085), abs=0.0005)


@pytest.mark.parametrize("interval", [wilson, clopper_pearson])
def test_empty_denominator(interval):
    assert interval(0, 0) is None


@pytest.mark.parametrize("interval", [wilson, clopper_pearson])
def test_invalid_count(interval):
    with pytest.raises(ValueError):
        interval(2, 1)


def test_exact_zero_events_two_sided():
    assert clopper_pearson(0, 20) == pytest.approx((0.0, 1 - 0.025 ** (1 / 20)))


def test_exact_all_events_two_sided():
    assert clopper_pearson(20, 20) == pytest.approx((0.025 ** (1 / 20), 1.0))
