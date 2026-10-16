"""Evaluate the existing sklearn forest on CUDA without retraining its trees.

Keep the original split thresholds, leaf probabilities, and averaging rule.
Scores at the decision boundary are confirmed by sklearn to preserve ties.
"""
import logging
import numpy as np


class ForestPredictor:
    def __init__(self, forest, device):
        import torch
        self.torch, self.device, self.forest = torch, device, forest
        self.positive = list(forest.classes_).index(1)
        self.depth = max(tree.tree_.max_depth for tree in forest.estimators_)
        offsets = np.cumsum([0]+[tree.tree_.node_count for tree in forest.estimators_[:-1]])
        left, right, features, thresholds, probabilities = [], [], [], [], []
        for offset, estimator in zip(offsets, forest.estimators_):
            tree = estimator.tree_
            ids = np.arange(tree.node_count)+offset
            left.append(np.where(tree.children_left>=0,tree.children_left+offset,ids))
            right.append(np.where(tree.children_right>=0,tree.children_right+offset,ids))
            features.append(np.maximum(tree.feature,0))
            thresholds.append(tree.threshold)
            values = tree.value[:,0,:]
            probabilities.append(values[:,self.positive]/np.maximum(values.sum(axis=1),1e-300))
        def tensor(parts):return torch.as_tensor(np.concatenate(parts),device=device)
        self.roots = torch.as_tensor(offsets,device=device)
        self.left,self.right,self.features = tensor(left),tensor(right),tensor(features)
        self.thresholds,self.probabilities = tensor(thresholds),tensor(probabilities)

    def predict(self, values, cut):
        t=self.torch
        results=[]
        with t.no_grad():
            for start in range(0,len(values),32768):
                current=values[start:start+32768]
                x=t.as_tensor(current,device=self.device,dtype=t.float64)
                rows=t.arange(len(x),device=self.device)[:,None]
                nodes=self.roots[None,:].expand(len(x),-1)
                for _ in range(self.depth):
                    below=x[rows,self.features[nodes]]<=self.thresholds[nodes]
                    nodes=t.where(below,self.left[nodes],self.right[nodes])
                scores=self.probabilities[nodes].mean(dim=1).cpu().numpy()
                # Summation order can differ by a few ulps. Resolve these rare
                # decisions with the original estimator, rather than move a cut.
                ties=np.abs(scores-cut)<1e-10
                if ties.any():scores[ties]=self.forest.predict_proba(current[ties])[:,self.positive]
                results.append(scores>=cut)
        return np.concatenate(results)


def accelerator(forest, device):
    if not device or not str(device).startswith('cuda'):
        return None
    try:
        return ForestPredictor(forest,device)
    except (ImportError,RuntimeError,MemoryError) as exc:
        logging.warning('Artifact forest remains on CPU: %s',exc)
        return None
