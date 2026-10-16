"""Studio's Consensus ML runs classic's engine as a job; four new columns wait beside the table."""

import json

import numpy as np
import pandas as pd
import pytest

from spacemap import dataio
from spacemap_gui.studio import consensus, tasks
from test_studio_leiden import project  # noqa: F401  (fixture: CD3, PanCK and CD68 populations in R1 and R2)

MARKERS = ["CD3", "PanCK", "CD68", "CD20"]


@pytest.fixture
def voted(project):  # noqa: F811
    """Three methods' phenotypes: right for most cells, each wrong on a different tenth."""
    root, path = project
    frame = pd.read_csv(path)
    truth = np.array(["T cell", "Tumour", "Macrophage"])[np.argmax(frame[["CD3", "PanCK", "CD68"]].to_numpy(), axis=1)]
    rng = np.random.default_rng(5)
    for method in ("phenotype_a", "phenotype_b", "phenotype_c"):
        calls = truth.copy()
        wrong = rng.random(len(calls)) < 0.1
        calls[wrong] = rng.choice(["T cell", "Tumour", "Macrophage"], wrong.sum())
        frame[method] = calls
    frame.to_csv(path, index=False)
    return root, path, truth


def job(root, path, **changes):
    chosen = {
        **consensus.values(root),
        "markers": MARKERS,
        "phenotyping_methods": ["phenotype_a", "phenotype_b", "phenotype_c"],
        "min_consensus_methods": 3,
        "n_estimators": 30,
        "min_training_cells": 20,
        "enable_filter": False,
        "exclude_phenotypes": [],
        **changes,
    }
    output = consensus.free_name(chosen["output_phenotype_col"], set(dataio.columns(path)))
    settings = {
        "table": str(path),
        "markers": MARKERS,
        "output_col": output,
        "cell_id_col": "cell_id",
        "project_root": str(root),
        "config": chosen,
    }
    settings["revision"] = tasks.source_revision(settings)
    return settings


def test_consensus_labels_every_cell_from_the_agreed_ones(voted, tmp_path, monkeypatch):
    from spacemap import ml_consensus
    progress = ml_consensus.operation_progress
    intervals = []
    def check_progress(notify, description, interval=None):
        intervals.append(interval)
        return progress(notify, description, interval=interval)
    monkeypatch.setattr(ml_consensus, 'operation_progress', check_progress)
    root, path, truth = voted
    messages = []
    found = consensus.run(job(root, path), tmp_path / "job", lambda message, *_: messages.append(message))
    assert len(intervals) >= 4 and all(interval is None for interval in intervals)
    assert not any('s elapsed' in message for message in messages)
    json.dumps(found, allow_nan=False)  # the Jobs list sends it to the page as strict JSON
    output = found["column"]
    assert output == "phenotype_ml_consensus" and found["columns"] == [
        output,
        output + "_score",
        output + "_consensus",
        output + "_refined",
    ]
    table = dataio.load(path, columns=[output, output + "_consensus", output + "_score"])
    assert (table[output].to_numpy() == truth).mean() > 0.95  # the model fixes most single-method errors
    assert 0.5 < found["consensus_share"] < 0.95 and table[output + "_score"].between(0, 1).all()
    assert output not in pd.read_csv(path, nrows=0).columns  # the CSV itself is untouched
    assert {item["name"] for item in found["phenotypes"]} == {"T cell", "Tumour", "Macrophage"}
    assert found["profile"]["rows"] and "validation_accuracy" in found["report"]
    # A second run never replaces the first one's columns.
    assert consensus.free_name("phenotype_ml_consensus", set(dataio.columns(path))) == "phenotype_ml_consensus_2"


def test_settings_are_tab7s_file_and_runs_are_checked(voted):
    from spacemap_gui.studio.settings import revision

    root, path, _ = voted
    saved = consensus.save(
        root,
        {"phenotyping_methods": ["phenotype_a", "phenotype_b"], "min_consensus_methods": 2,
         "cpu_workers": 12, "execution_backend": "auto", "gpu_bins": 512},
        revision(consensus.values(root)),
    )
    assert saved["cpu_workers"] == 12 and saved["execution_backend"] == "auto" and saved["gpu_bins"] == 512
    assert (
        json.loads(consensus.cache_path(root).read_text())["min_consensus_methods"] == 2
        and saved["n_estimators"] == 1500
    )
    for wrong in ({"ml_method": "magic"}, {"min_consensus_methods": 1}, {"output_phenotype_col": "x"}, {"colour": 1}):
        with pytest.raises(ValueError):
            consensus.check(wrong)
    with pytest.raises(ValueError, match="two phenotyping methods"):
        consensus.check_run(
            {**consensus.values(root), "phenotyping_methods": ["phenotype_a"], "markers": MARKERS}, dataio.columns(path)
        )
    with pytest.raises(ValueError, match="between 2 and 2"):
        consensus.check_run(
            {**consensus.values(root), "min_consensus_methods": 3, "markers": MARKERS}, dataio.columns(path)
        )


def test_configured_feature_columns_loaded_and_filter_not_silently_ignored(voted, tmp_path, monkeypatch):
    from spacemap import ml_consensus
    root, path, _ = voted
    frame = pd.read_csv(path)
    frame['area'] = np.arange(len(frame)) + 5
    frame['tumor'] = np.arange(len(frame)) % 2
    frame.to_csv(path, index=False)
    captured = []
    prepare = ml_consensus.prepare_training_data
    def inspect(*args, **kwargs):
        result = prepare(*args, **kwargs)
        captured.extend(result[2])
        return result
    monkeypatch.setattr(ml_consensus, 'prepare_training_data', inspect)
    settings = job(root, path, morphology_cols=['area'], compartment_cols=['tumor'],
                   feature_weights={'markers': 1, 'morphology': 1, 'compartments': 1})
    consensus.run(settings, tmp_path / 'job', lambda *args: None)
    assert any('area' in name for name in captured) and any('tumor' in name for name in captured)
    settings['config'].update(enable_filter=True, filter_col='missing_tissue')
    with pytest.raises(ValueError, match='missing_tissue'):
        consensus.check_run(settings['config'], dataio.columns(path))


def test_real_worker_uses_requested_cpu_budget_and_logs_once(voted):
    from spacemap_gui.studio import jobs
    from test_studio_jobs import wait_job
    from spacemap.consensus_execution import cpu_budget
    root, path, _ = voted
    settings = job(root, path, cpu_workers=4)
    started = jobs.start(root, 'consensus_ml', settings)
    result = wait_job(root, started['id'])
    log = jobs.log(root, started['id'])
    assert result['status'] == 'done', log
    assert result['result']['execution']['workers'] == cpu_budget(4)['workers']
    folder = jobs.directory(root, started['id'])
    recorded = json.loads((folder / 'report/execution.json').read_text())
    assert recorded['cpu_workers'] == cpu_budget(4)['workers']
    assert set(recorded['stage_seconds']) == {'validation', 'final_fit', 'prediction'}
    assert log.count('STARTING ML CONSENSUS ANALYSIS') == 1
    assert 'Validation fit:' in log and 'Final fit:' in log


def test_bad_cpu_budget_does_not_leave_an_active_job(tmp_path):
    from spacemap_gui.studio import jobs
    with pytest.raises(ValueError, match='CPU cores'):
        jobs.start(tmp_path, 'consensus_ml', {'config': {'cpu_workers': -1}})
    assert jobs.recent(tmp_path) == []
