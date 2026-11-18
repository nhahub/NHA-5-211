"""Tests for Streamlit dashboard components and graceful error handling."""

from __future__ import annotations

from pathlib import Path

from dashboard.components.cards import threat_badge_html, threat_badge_text
from dashboard.components.charts import (
    plot_attack_class_donut,
    plot_attack_port_heatmap,
    plot_confidence_histogram,
    plot_threat_level_donut,
    plot_timeline_area,
    plot_top_attacking_ips,
    plot_top_destination_ports,
)
from dashboard.data_access import generate_demo_logs


def test_dashboard_charts_and_badges_render_cleanly() -> None:
    df_demo = generate_demo_logs(n_rows=80, seed=42)
    assert len(df_demo) == 80

    assert "threat-critical" in threat_badge_html("Critical")
    assert "Critical" in threat_badge_text("Critical")

    for fig in (
        plot_attack_class_donut(df_demo),
        plot_threat_level_donut(df_demo),
        plot_timeline_area(df_demo),
        plot_confidence_histogram(df_demo),
        plot_top_attacking_ips(df_demo),
        plot_top_destination_ports(df_demo),
        plot_attack_port_heatmap(df_demo),
    ):
        assert fig is not None
        assert len(fig.data) >= 1


def test_streamlit_apptest_smoke_with_existing_artifacts() -> None:
    from streamlit.testing.v1 import AppTest

    app_path = Path(__file__).resolve().parent.parent / "app.py"
    at = AppTest.from_file(str(app_path), default_timeout=90)
    at.run()
    assert not at.exception
    assert len(at.tabs) >= 7


def test_streamlit_apptest_graceful_degradation_without_models(tmp_path: Path, monkeypatch) -> None:
    """Dashboard degrades cleanly with a clear banner when models/ is empty or LFS pointers."""
    from streamlit.testing.v1 import AppTest

    from dashboard.data_access import build_test_detection_logs, load_cached_artifacts
    from src.config import get_app_config

    empty_models = tmp_path / "empty_models"
    empty_models.mkdir(parents=True, exist_ok=True)
    # Place an unfetched Git LFS pointer stub to simulate Docker/clone without git lfs pull
    (empty_models / "best_mlp.keras").write_text(
        "version https://git-lfs.github.com/spec/v1\noid sha256:0000\nsize 1234\n",
        encoding="utf-8",
    )

    cfg_copy = get_app_config().model_copy(deep=True)
    cfg_copy.paths.models_dir = str(empty_models)
    monkeypatch.setattr("dashboard.data_access.get_app_config", lambda *args, **kwargs: cfg_copy)
    load_cached_artifacts.clear()
    build_test_detection_logs.clear()

    try:
        app_path = Path(__file__).resolve().parent.parent / "app.py"
        at = AppTest.from_file(str(app_path), default_timeout=90)
        at.run()
        assert not at.exception
        assert len(at.tabs) >= 7
        warnings_text = " ".join(str(w.value) for w in at.warning)
        assert "Git LFS" in warnings_text or "Absent" in warnings_text
    finally:
        load_cached_artifacts.clear()
        build_test_detection_logs.clear()
