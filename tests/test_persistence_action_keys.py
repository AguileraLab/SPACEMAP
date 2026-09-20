from __future__ import annotations

import pytest

from spacemap_gui.persistence import is_transient_key, update_stored_config_key


@pytest.mark.parametrize(
    "key",
    [
        "tab9_marker_single_hist_download_png",
        "tab9_marker_single_hist_download_pdf",
        "tab9_marker_single_hist_download_svg",
        "tab9_marker_single_hist_png_download",
        "tab9_marker_single_hist_pdf_download",
        "tab9_marker_single_hist_svg_download",
    ],
)
def test_marker_analysis_download_state_is_never_persisted(key: str) -> None:
    assert is_transient_key(key)

    config = {"tab_configurations": {}}
    update_stored_config_key(config, key, False)
    assert key not in str(config)


def test_marker_analysis_figure_settings_remain_persistent() -> None:
    assert not is_transient_key("tab9_marker_single_hist_dpi")
    assert not is_transient_key("tab9_marker_single_hist_width")


@pytest.mark.parametrize(
    "key",
    [
        "tools_mr_vv_results",
        "tools_mr_vv_seen_keys",
        "tools_mr_vv_fetch_more",
        "tools_mr_vv_last_file",
        "tools_mr_vv_assign_pheno_cell_1",
        "tools_mr_vv_assign_save_cell_1",
    ],
)
def test_manual_reference_runtime_state_is_never_persisted(key: str) -> None:
    assert is_transient_key(key)
