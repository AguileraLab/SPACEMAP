"""Cluster summaries on values already transformed by the analysis pipeline."""


def standardize_groups(summary):
    """Each marker compared across groups, using classic's sample standard deviation."""
    spread = summary.std(axis=0).replace(0, 1).fillna(1)
    return ((summary - summary.mean(axis=0)) / spread).fillna(0)


def summarize(frame, group, columns, aggregation="median"):
    if aggregation not in {"median", "mean"}:
        raise ValueError("Choose Median or Mean for the heatmap.")
    summary = frame.groupby(group)[columns].agg(aggregation)
    return summary, standardize_groups(summary)


def named_profiles(matrix, names):
    """Classic publication convention: equal-weight mean of cluster z-profiles sharing a name."""
    labels = [names.get(row, names.get(str(row), "Unassigned")) for row in matrix.index]
    return matrix.groupby(labels, sort=True).mean()
