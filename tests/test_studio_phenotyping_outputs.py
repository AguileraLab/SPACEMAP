"""Visible settings reach runs; replacing results is explicit across phenotyping."""
import json

import numpy as np
import pandas as pd
import pytest

from spacemap import dataio, labels
from spacemap_gui.studio import clustering, consensus, jobs, leiden, subphenotypes, tasks
from spacemap_gui.studio.settings import revision
from test_studio_commands import run, setup, saved_resolve_workspace  # noqa: F401


@pytest.mark.parametrize("method", ["leiden", "som", "vortex", "consensus", "subphenotyping", "resolve", "mr_ml"])
def test_every_method_requires_overwrite_and_keeps_exact_output(setup, monkeypatch, method):  # noqa: F811
    _, table, root, _ = setup
    frame = pd.read_csv(table)
    frame["cell_id"] = range(len(frame))
    # Consensus defaults to tissue-only analysis; this fixture represents
    # already selected tissue so the test can reach the overwrite guard.
    frame["tissue"] = 1
    frame["phenotype_a"], frame["phenotype_b"] = "A", "A"
    output = "phenotype_test"
    clustered = method in {"leiden", "som", "vortex", "subphenotyping"}
    outputs = [output, output + "_clusters"] if clustered else [output]
    if method == "consensus":
        outputs += [output + s for s in ("_score", "_refined", "_consensus")]
    if method == "mr_ml":
        outputs += [output + s for s in ("_confidence", "_refined")]
    for column in outputs:
        frame[column] = "Earlier result"
    frame.to_csv(table, index=False)
    dataio.ensure_fast_copy(table, force=True)
    flags = dict(norm_sinh=False, norm_z1=True, norm_z2=True, norm_cdf=True, norm_nlog=True, batch_id_col="")
    config = {"markers": ["M0", "M1"], "output_col": output, **flags}
    args = {}
    if method == "leiden":
        path = leiden.cache_path(root)
        get_values = lambda: leiden.values(root)
    elif method in {"som", "vortex"}:
        path = clustering.cache_path(root, method)
        get_values = lambda: clustering.values(root, method)
    elif method == "consensus":
        path = consensus.cache_path(root)
        config = {"markers": ["M0", "M1"], "phenotyping_methods": ["phenotype_a", "phenotype_b"],
                  "min_consensus_methods": 2, "output_phenotype_col": output}
        get_values = lambda: consensus.values(root)
    elif method == "subphenotyping":
        path = subphenotypes.cache_path(root)
        config = {"markers": ["M0", "M1"], "parent_col": "phenotype_a", "parent_values": ["A"], "studio_output_col": output}
        get_values = lambda: subphenotypes.values(root)
    else:
        path = None
        args = {"column": output, **({"reference_column": "phenotype_ref1"} if method == "mr_ml" else {})}
    if path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(config))
        args["settings_revision"] = revision(get_values())
    command = {"consensus": "run_consensus_ml", "resolve": "run_resolve_all", "mr_ml": "run_mr_ml"}.get(method, "run_" + method)
    if method == "resolve":
        saved_resolve_workspace(setup)
    captured = []
    def start(root, kind, settings, **kw):
        captured.append(settings)
        return {"id": "a"*32, "kind": kind, "status": "starting"}
    monkeypatch.setattr(jobs, "start", start)
    rejected = run(setup, command, args, confirmed=True)
    assert rejected["status"] == "error" and "Overwrite existing results" in rejected["message"], rejected
    assert not captured
    if path:
        stale = run(setup, command, {**args, "overwrite": True, "settings_revision": "stale"}, confirmed=True)
        assert stale["status"] == "error" and "settings changed" in stale["message"], stale
        assert not captured
    accepted = run(setup, command, {**args, "overwrite": True}, confirmed=True)
    assert accepted["status"] == "done", accepted
    settings = captured[-1]
    assert set(settings["overwrite_columns"]) == set(outputs)
    assert settings["output_col"] == output + ("_clusters" if clustered else "")
    if method in {"leiden", "som", "vortex"}:
        used = settings.get("pipeline", settings.get("config"))
        assert all(used[k] == v for k, v in flags.items())
    columns = {c: [0.8]*len(frame) if c.endswith(("_score", "_confidence")) else ["New"]*len(frame) for c in outputs}
    # Replace sidecar overrides as well as saved columns, retaining unrelated reference edits.
    labels.append(table, [{"row": 0, "column": output, "value": "Old edit"},
                          {"row": 1, "column": "phenotype_ref1", "value": "Manual"}])
    settings["revision"] = tasks.source_revision(settings)
    tasks.save_result(settings, columns, frame.cell_id, lambda *args: None)
    found = dataio.load(table)
    assert found[output].eq("New").all()
    assert found.phenotype_ref1.iloc[1] == "Manual"
    pd.testing.assert_series_equal(found.M0, frame.M0)
    assert output + "_2" not in found


@pytest.mark.parametrize("method", ["som", "vortex"])
def test_corrected_inputs_follow_every_selected_transform(method):
    from scipy.special import ndtr
    from spacemap_workflows import som_pipeline
    frame = pd.DataFrame({"A_corr": [1., 2., 7., 3.], "B_corr": [4., 2., 6., 1.], "C_corr": [3., 5., 1., 9.]})
    markers = list(frame)
    config = {**som_pipeline.DEFAULTS, "norm_sinh": False, "norm_z1": True, "norm_z2": True,
              "norm_cdf": True, "norm_nlog": True, "batch_id_col": ""}
    # Pipeline preprocessing is shared by SOM and VORTEX; compare with independent arithmetic.
    normalise = som_pipeline.normalise
    if method == "vortex":
        from spacemap import ph_vortex_utils
        normalise = lambda df, cfg, names, cof: (ph_vortex_utils.apply_flexible_normalization(
            df, names, [], "_norm", apply_z1=cfg["norm_z1"], apply_z2=cfg["norm_z2"],
            apply_cdf=cfg["norm_cdf"], apply_nlog=cfg["norm_nlog"]), [m + "_norm" for m in names])
    result, columns = normalise(frame, config, markers, 5.)
    values = frame.to_numpy()
    z1 = (values - values.mean(axis=0)) / values.std(axis=0)
    z2 = (z1 - z1.mean(axis=1, keepdims=True)) / z1.std(axis=1, keepdims=True)
    expected = -np.log(np.clip(1 - ndtr(z2), 1e-10, 1))
    np.testing.assert_allclose(result[columns], expected, rtol=1e-5, atol=1e-6)
    if method == "som":
        pd.testing.assert_frame_equal(result[markers], frame)


@pytest.mark.parametrize("inverse", [False, True])
def test_no_selected_normalization_keeps_values_in_the_requested_scale(inverse):
    from spacemap_workflows.som_pipeline import DEFAULTS, normalise
    frame = pd.DataFrame({"A_corr": [1., 2., 3.], "B": [4., 5., 6.]})
    config = {**DEFAULTS, "norm_sinh": inverse, "norm_z1": False, "norm_z2": False, "norm_cdf": False, "norm_nlog": False}
    result, columns = normalise(frame, config, list(frame), 5.)
    expected = frame.copy()
    if inverse:
        expected["A_corr"] = np.sinh(expected.A_corr) * 5
    np.testing.assert_allclose(result[columns], expected)
    pd.testing.assert_frame_equal(result[list(frame)], frame)
