from ball_simulator.regime_inspection import (
    TrajectorySummary,
    _contact_episode_count,
    select_stratified_regimes,
)
import numpy as np


def test_contact_episode_count():
    assert _contact_episode_count(np.array([0, 1, 1, 0, 1, 0], dtype=bool)) == 2
    assert _contact_episode_count(np.zeros(5, dtype=bool)) == 0


def make_summary(index: int) -> TrajectorySummary:
    x = float(index + 1)
    return TrajectorySummary(
        trajectory_id=f"{index:08d}", initial_normal_speed=x,
        initial_tangential_speed=11.0-x, initial_linear_speed=12.0,
        initial_angular_speed=10.0*x, incidence_cosine=x/11.0,
        friction=0.05*x, normal_stiffness=1e4*x, normal_damping=0.05*x,
        mass=0.2, radius=0.05, contact_episodes=2 if index == 9 else 1,
        has_sticking=index % 2 == 0, has_sliding=index % 2 == 1,
        maximum_penetration=1e-4*x, peak_normal_force=100.0*x,
        final_linear_speed=5.0, final_angular_speed=20.0,
    )


def test_selector_finds_key_regimes_without_reuse():
    selected = select_stratified_regimes([make_summary(i) for i in range(20)])
    names = {item.regime for item in selected}
    ids = [item.trajectory_id for item in selected]
    assert "representative" in names
    assert "fast-impact" in names
    assert "grazing-impact" in names
    assert "repeated-contact" in names
    assert len(ids) == len(set(ids))
