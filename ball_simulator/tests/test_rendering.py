from ball_simulator.rendering.jobs import render_is_complete

def test_completed_render_is_detected(
    tmp_path,
):
    trajectory = tmp_path / (
        "trajectory_00000000"
    )
    trajectory.mkdir()
    (trajectory / "_SUCCESS").write_text(
        "{}",
        encoding="utf-8",
    )

    assert render_is_complete(
        tmp_path,
        "00000000",
    )