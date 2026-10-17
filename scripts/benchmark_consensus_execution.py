"""Bounded synthetic benchmark; never opens project data or starts Studio jobs.

Run with native numerical threads limited to one, as in the Studio worker.
"""

import argparse
import hashlib
import json
from pathlib import Path
import platform
import statistics
import time

import numpy as np
import pandas as pd
import sklearn
from sklearn.datasets import make_classification
from sklearn.preprocessing import LabelEncoder, StandardScaler

from spacemap.consensus_execution import cpu_budget, gpu_probe
from spacemap.ml_consensus import train_final_model, predict_non_consensus


def digest(model):
    hashes = []
    for tree in model.estimators_:
        h = hashlib.sha256(str(tree.random_state).encode())
        # Avoid structured-array padding bytes when comparing tree structures.
        for value in (tree.tree_.children_left, tree.tree_.children_right, tree.tree_.feature,
                      tree.tree_.threshold, tree.tree_.value, tree.tree_.impurity,
                      tree.tree_.n_node_samples, tree.tree_.weighted_n_node_samples):
            h.update(value.tobytes())
        hashes.append(h.hexdigest())
    return hashes


def run():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    X, y = make_classification(n_samples=16000, n_features=22, n_informative=16,
                               n_redundant=2, n_classes=12, random_state=42)
    train = pd.DataFrame(X[:6000])
    test = pd.DataFrame(X[6000:])
    labels = pd.Series(y[:6000]).astype(str)
    encoder = LabelEncoder().fit(labels)
    scaler = StandardScaler().fit(train)
    report = {'host': platform.node(), 'sklearn': sklearn.__version__, 'training_rows': len(train),
              'prediction_rows': len(test), 'features': 22, 'trees': 1500,
              'random_state': 42, 'class_weight': 'balanced', 'fits': [], 'gpu': gpu_probe()}
    base_hash = base_labels = base_probability = None
    for requested in (2, 8, 16):
        workers = cpu_budget(requested)['workers']
        start = time.perf_counter()
        model = train_final_model(train, labels, encoder, scaler, 1500, 42, cpu_workers=workers)
        elapsed = time.perf_counter() - start
        hashes = digest(model)
        prediction, probability = predict_non_consensus(model, scaler, test, encoder)
        if base_hash is None:
            base_hash, base_labels, base_probability = hashes, prediction, probability
        report['fits'].append({'workers': workers, 'seconds': elapsed,
                               'same_trees': hashes == base_hash,
                               'same_labels': bool(np.array_equal(prediction, base_labels)),
                               'max_confidence_delta': float(np.max(np.abs(probability - base_probability)))})
        if requested == 8:
            timings = {'two_pass': [], 'one_pass': []}
            expected = None
            for _ in range(3):
                start = time.perf_counter()
                scaled = scaler.transform(test)
                expected = encoder.inverse_transform(model.predict(scaled))
                confidence = model.predict_proba(scaled).max(axis=1)
                timings['two_pass'].append(time.perf_counter() - start)
                start = time.perf_counter()
                actual, actual_confidence = predict_non_consensus(model, scaler, test, encoder)
                timings['one_pass'].append(time.perf_counter() - start)
                assert np.array_equal(actual, expected)
                np.testing.assert_allclose(actual_confidence, confidence, atol=1e-12)
            report['prediction_seconds'] = {k: statistics.median(v) for k, v in timings.items()}
        del model
    args.output.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    run()
