import pytest
from src.training.trainer import EarlyStopping


def test_plateau_resume_and_cumulative_improvement():
    cfg = {"enabled": True, "patience": 3, "min_delta": 0.01}
    first = EarlyStopping(cfg)
    assert not first.update(3.0)
    assert not first.update(2.996)
    resumed = EarlyStopping(cfg)
    resumed.load_state_dict(first.state_dict())
    for loss in (2.992, 2.985, 2.984, 2.983):
        assert first.update(loss) == resumed.update(loss) == False
    assert first.update(2.982) == resumed.update(2.982) == True


def test_disabled_and_invalid_loss():
    stopper = EarlyStopping({"enabled": False, "patience": 1})
    assert not stopper.update(3.0)
    assert not stopper.update(4.0)
    with pytest.raises(FloatingPointError):
        stopper.update(float("nan"))


def test_zero_delta_ties_do_not_reset_patience():
    stopper = EarlyStopping({"enabled": True, "patience": 1, "min_delta": 0})
    assert not stopper.update(3.0)
    assert stopper.update(3.0)
