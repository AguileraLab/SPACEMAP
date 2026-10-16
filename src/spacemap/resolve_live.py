"""RESOLVE on one ROI's cells, fast enough to re-run while a threshold is dragged.

This reproduces Tab 7's RESOLVE with predefined thresholds, step for step, on
arrays instead of row by row. ``tests/test_resolve_live.py`` checks that it
gives the same phenotype as Tab 7's own functions for every cell.

Tab 7's steps, in order (``tabs/tab7_phenotyping/resolve_clustering.py``, run
section, and ``spacemap.ph_thres_utils``):

1. Cells with a missing value in any selected marker are left out.
2. **Positive** where ``raw - threshold > 0`` (``_apply_thresholds_from_table``).
3. **Tie score** per marker: the distance above the threshold after the
   configured transform (identity, ``asinh(x / cofactor)`` or ``log1p``),
   clipped at 0, then normalised per batch in the configured order:
   Z1 (per marker), Z2 (per cell), CDF, -log(1 - x); invalid values become 0
   (``apply_flexible_normalization``). Here the batch is the ROI.
4. **Criteria:** a phenotype fires when all markers of any one of its
   criteria are positive (``set_classification_flags``). Criteria naming a
   marker that is not selected are ignored.
5. **Hierarchy:** a phenotype whose markers are a strict subset of another's
   is its parent; a parent is dropped where its child fires
   (``prune_parent_categories``).
6. **Tie-break:** when several phenotypes remain, the one whose markers have
   the highest tie score wins; on an exact tie, the first in panel order
   (``_assign_single_row``). The score is over all the phenotype's markers:
   Tab 7's "unique markers" step removes nothing, and this matches Tab 7.
7. **Supportive markers** settle contests between phenotypes whose criteria
   are met (``apply_supportive_markers``).
8. **Must not have:** a call is moved when a forbidden marker's tie score is
   higher than the best of the phenotype's own markers, to the phenotype that
   requires that marker, or to "Others" (``apply_exclusion_markers``).
9. ``unclassified`` becomes the configured label (Others).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from spacemap.ph_thres_utils import NORMALIZATION_EPSILON as EPSILON


@dataclass
class Panel:
    """RESOLVE's phenotype panel: criteria (OR of ANDs), supportive and forbidden markers."""

    definitions: dict[str, list[list[str]]]
    supportive: dict[str, list[str]] = field(default_factory=dict)
    exclusions: dict[str, list[str]] = field(default_factory=dict)
    unclassified_label: str = "Others"

    @classmethod
    def from_payload(cls, payload: dict, unclassified_label: str = "Others") -> "Panel":
        """From the phenotype panel file Tab 7 saves (``resolve_pheno_*.json``)."""
        criteria_cols = list(payload.get("criteria_cols") or [])
        definitions, supportive, exclusions = {}, {}, {}
        for row in payload.get("table") or []:
            name = str(row.get("Phenotype") or "").strip()
            if not name:
                continue
            criteria = [list(row.get(c) or []) for c in criteria_cols if isinstance(row.get(c), list) and row.get(c)]
            if criteria:
                definitions[name] = criteria
            if isinstance(row.get("Supportive"), list) and row["Supportive"]:
                supportive[name] = list(row["Supportive"])
            if isinstance(row.get("Must not have"), list) and row["Must not have"]:
                exclusions[name] = list(row["Must not have"])
        return cls(definitions, supportive, exclusions, unclassified_label)


@dataclass
class Settings:
    """How tie scores are made (Tab 7's RESOLVE configuration)."""

    norm_flags: tuple[bool, bool, bool, bool] = (True, True, True, True)  # Z1, Z2, CDF, NLog
    norm_order: tuple[str, ...] = ("Z1", "Z2", "CDF", "NLog")
    tie_score_transform: str = "identity_residual"
    tie_score_cofactor: float = 5.0

    @classmethod
    def from_config(cls, config: dict) -> "Settings":
        return cls(
            norm_flags=(
                bool(config.get("norm_z1", True)),
                bool(config.get("norm_z2", True)),
                bool(config.get("norm_cdf", True)),
                bool(config.get("norm_nlog", True)),
            ),
            norm_order=tuple(config.get("norm_order") or ("Z1", "Z2", "CDF", "NLog")),
            tie_score_transform=str(config.get("tie_score_transform") or "identity_residual"),
            tie_score_cofactor=float(config.get("tie_score_cofactor", 5.0)),
        )


def hierarchy(definitions: dict[str, list[list[str]]]) -> dict[str, str | None]:
    """Tab 7's automatic parent/child: the largest phenotype whose markers are a strict subset."""
    marker_sets = {name: {m for clause in groups for m in clause} for name, groups in definitions.items()}
    result: dict[str, str | None] = {}
    for child, child_markers in marker_sets.items():
        best, best_size = None, 0
        for parent, parent_markers in marker_sets.items():
            if parent != child and parent_markers < child_markers and len(parent_markers) > best_size:
                best, best_size = parent, len(parent_markers)
        result[child] = best
    return result


def _transform(values: np.ndarray, kind: str, cofactor: float) -> np.ndarray:
    from spacemap.ph_thres_utils import _canonical_tie_score_transform

    kind = _canonical_tie_score_transform(kind)
    if kind == "asinh_difference":
        result = values / cofactor  # keeps the input's layout
        return np.arcsinh(result, out=result)
    if kind == "log1p_difference":
        result = np.clip(values, 0, None)
        return np.log1p(result, out=result)
    return values


def _steps(settings: Settings) -> list[str]:
    z1, z2, cdf, nlog = settings.norm_flags
    enabled = {"Z1": z1, "Z2": z2, "CDF": cdf, "NLog": nlog}
    return [step for step in (str(s).strip() for s in settings.norm_order) if enabled.get(step)]


def _normalise(data: np.ndarray, steps: list[str], column_stats=None) -> np.ndarray:
    """apply_flexible_normalization on one batch, as arrays (no missing values here).

    *column_stats* ((means, stds) of the whole batch) lets a subset of cells be
    normalised exactly as in a whole-batch run when Z1 comes first.
    """
    from scipy.special import ndtr

    data = np.array(data, dtype=float)
    for position, step in enumerate(steps):
        if step == "Z1":
            if column_stats is not None and position == 0:
                means, stds = column_stats
            else:
                means, stds = data.mean(axis=0), data.std(axis=0)
                stds[stds == 0] = EPSILON
            data -= means
            data /= stds
        elif step == "Z2":
            means = data.mean(axis=1, keepdims=True)
            stds = data.std(axis=1, keepdims=True)
            stds[stds == 0] = EPSILON
            data -= means
            data /= stds
        elif step == "CDF":
            data = ndtr(data)
        elif step == "NLog":
            data = -np.log(np.maximum(1 - data, EPSILON))
    return np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0, copy=False)


@dataclass
class Result:
    phenotypes: np.ndarray  # one name per cell (object array); None for cells left out
    counts: dict[str, int]
    positive: np.ndarray  # cells x markers, bool (the engine's own copy: valid until the next assign)
    fired: np.ndarray  # cells x phenotypes, criteria met (before hierarchy)
    kept: np.ndarray  # cells x phenotypes, after hierarchy
    score_rows: object = None  # tie scores for any rows, as a whole-ROI run would give them
    codes: np.ndarray | None = None  # per assigned cell, an index into names (fast colouring)
    names: list | None = None  # the name of each code


class LiveResolve:
    """RESOLVE for one ROI: set up once from its raw marker values, then assign for any thresholds."""

    def __init__(self, values: np.ndarray, markers: list[str], panel: Panel, settings: Settings | None = None):
        self.markers = list(markers)
        self.panel = panel
        self.settings = settings or Settings()
        values = np.asarray(values, dtype=float)
        self.valid = np.isfinite(values).all(axis=1)  # step 1
        # Kept by column (one marker's cells contiguous), as a threshold change works per marker.
        self.values = np.asfortranarray(values if self.valid.all() else values[self.valid])
        self.index = {m: i for i, m in enumerate(self.markers)}
        self.names = list(panel.definitions)
        self.hierarchy = hierarchy(panel.definitions)
        self._transformed = _transform(self.values, self.settings.tie_score_transform, self.settings.tie_score_cofactor)
        # Criteria as marker-index lists; a criterion naming an unselected marker is ignored.
        self._criteria = [
            [[self.index[m] for m in clause] for clause in groups if all(m in self.index for m in clause)]
            for groups in panel.definitions.values()
        ]
        self._own = [
            sorted({self.index[m] for clause in groups for m in clause if m in self.index})
            for groups in panel.definitions.values()
        ]
        # Dragging a threshold changes one marker: keep what depends on each marker,
        # by columns (contiguous), and redo only the changed ones. On PDAC's largest
        # ROI (1.38 million cells) a full recompute took seconds per change.
        self._values_columns = self.values
        self._transformed_columns = np.asfortranarray(self._transformed)  # already so: no copy
        self._uses = [{i for clause in groups for i in clause} for groups in self._criteria]
        self._last: np.ndarray | None = None
        self._positive = np.zeros(self.values.shape, dtype=bool, order="F")
        self._fired = np.zeros((self.values.shape[0], len(self.names)), dtype=bool, order="F")
        self._column_stats: dict[tuple[int, float], tuple[float, float]] = {}

    def _update(self, t: np.ndarray) -> None:
        """Positive cells and criteria met, recomputed only for markers whose threshold changed."""
        changed = set(range(len(t))) if self._last is None else set(np.flatnonzero(t != self._last).tolist())
        if not changed:
            return
        for i in changed:
            self._positive[:, i] = (self._values_columns[:, i] - t[i]) > 0  # step 2
        for j, groups in enumerate(self._criteria):  # step 4
            if self._last is not None and not (self._uses[j] & changed):
                continue
            column = np.zeros(self._positive.shape[0], dtype=bool)
            for clause in groups:
                column |= self._positive[:, clause].all(axis=1)
            self._fired[:, j] = column
        self._last = t.copy()

    def _z1_stats(self, threshold_t: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Per-marker mean and std of the clipped distance over the whole ROI (cached per threshold)."""
        means, stds = np.empty(len(threshold_t)), np.empty(len(threshold_t))
        for i, value in enumerate(threshold_t):
            key = (i, float(value))
            if key not in self._column_stats:
                distance = np.clip(self._transformed_columns[:, i] - value, 0, None)
                std = distance.std()
                self._column_stats[key] = (distance.mean(), std if std != 0 else EPSILON)
                if len(self._column_stats) > 64 * len(threshold_t):
                    self._column_stats.pop(next(iter(self._column_stats)))
            means[i], stds[i] = self._column_stats[key]
        return means, stds

    def _scorer(self, thresholds: np.ndarray):
        """Tie scores for any rows, normalised as a whole-ROI run would normalise them."""
        threshold_t = _transform(
            thresholds[None, :], self.settings.tie_score_transform, self.settings.tie_score_cofactor
        )
        steps = _steps(self.settings)
        if steps and steps[0] != "Z1" and "Z1" in steps:
            # Z1 after a per-cell step needs every cell's values: score all rows at once.
            everything = _normalise(np.clip(self._transformed - threshold_t, 0, None), steps)
            return lambda rows: everything[rows]
        stats = self._z1_stats(threshold_t[0]) if steps and steps[0] == "Z1" else None
        return lambda rows: _normalise(np.clip(self._transformed[rows] - threshold_t, 0, None), steps, stats)

    def assign(self, thresholds: dict[str, float]) -> Result:
        missing = [m for m in self.markers if m not in thresholds]
        if missing:
            raise ValueError("No threshold for: " + ", ".join(missing))
        t = np.array([float(thresholds[m]) for m in self.markers])
        self._update(t)  # steps 2 and 4
        positive, fired = self._positive, self._fired
        score_rows = self._scorer(t)  # step 3 (on demand)
        p = len(self.names)
        kept = fired.copy()  # step 5
        for child, parent in self.hierarchy.items():
            if parent is not None:
                kept[kept[:, self.names.index(child)], self.names.index(parent)] = False
        labels = self.names + ["unclassified", "Others"]
        unclassified, others = p, p + 1
        call = np.where(kept.any(axis=1), np.argmax(kept, axis=1), unclassified)
        # Tie scores are needed only where phenotypes compete, or a forbidden marker may move the call.
        supportive = {
            name: [self.index[m] for m in markers if m in self.index] for name, markers in self.panel.supportive.items()
        }
        need = kept.sum(axis=1) > 1
        if supportive:
            need |= fired.sum(axis=1) > 1
        rows = np.flatnonzero(need)
        if rows.size:
            block = score_rows(rows)
            strength = np.column_stack(
                [block[:, own].max(axis=1) if own else np.zeros(rows.size) for own in self._own]
            )  # step 6
            ranked = np.where(kept[rows], strength, -np.inf)
            call[rows] = np.where(kept[rows].sum(axis=1) > 1, np.argmax(ranked, axis=1), call[rows])
            if supportive:  # step 7
                contested = fired[rows].sum(axis=1) > 1
                support = np.column_stack(
                    [
                        np.where(
                            fired[rows, j],
                            block[:, supportive[name]].max(axis=1) if supportive.get(name) else 0.0,
                            -np.inf,
                        )
                        for j, name in enumerate(self.names)
                    ]
                )
                best, best_at = support.max(axis=1), np.argmax(support, axis=1)
                current = call[rows]
                current_ok = np.zeros(rows.size, dtype=bool)
                classified = current < p
                current_ok[classified] = fired[rows[classified], current[classified]]
                move = contested & current_ok & np.isfinite(best) & (best > 0) & (best_at != current)
                call[rows] = np.where(move, best_at, current)
        if self.panel.exclusions:  # step 8
            owners = {}
            for name, groups in self.panel.definitions.items():
                for clause in groups:
                    for marker in clause:
                        owners.setdefault(marker, name)
            before = call.copy()
            for name, forbidden in self.panel.exclusions.items():
                forbidden = [m for m in (forbidden or []) if m in self.index]
                if not forbidden or name not in self.names:
                    continue
                j = self.names.index(name)
                target = np.flatnonzero(before == j)
                own = self._own[j]
                if not target.size or not own:
                    continue
                target_scores = score_rows(target)
                rival = target_scores[:, [self.index[m] for m in forbidden]]
                vetoed = rival.max(axis=1) > target_scores[:, own].max(axis=1)
                if vetoed.any():
                    winning = np.array(forbidden, dtype=object)[np.argmax(rival[vetoed], axis=1)]
                    call[target[vetoed]] = [self.names.index(owners[m]) if m in owners else others for m in winning]
        final = np.array(labels, dtype=object)
        final[unclassified] = self.panel.unclassified_label  # step 9
        full = np.full(self.valid.shape[0], None, dtype=object)
        full[self.valid] = final[call]
        tally = np.bincount(call, minlength=len(labels))
        counts: dict[str, int] = {}
        for code in np.flatnonzero(tally):
            counts[str(final[code])] = counts.get(str(final[code]), 0) + int(tally[code])
        return Result(full, counts, positive, fired, kept, score_rows, call, [str(n) for n in final])

    def explain(self, cell: int, thresholds: dict[str, float], result: Result | None = None) -> dict:
        """Why one cell (row number in the ROI) got its phenotype: markers, criteria, hierarchy, tie scores."""
        if not self.valid[cell]:
            return {"left_out": True, "reason": "A selected marker has no value for this cell."}
        result = result or self.assign(thresholds)
        row = int(np.cumsum(self.valid)[cell] - 1)  # position among the cells that were assigned
        row_scores = result.score_rows(np.array([row]))[0]
        markers = [
            {
                "marker": m,
                "value": float(self.values[row, i]),
                "threshold": float(thresholds[m]),
                "positive": bool(result.positive[row, i]),
                "score": float(row_scores[i]),
            }
            for i, m in enumerate(self.markers)
        ]
        phenotypes = []
        for j, (name, groups) in enumerate(self.panel.definitions.items()):
            criteria = [
                {
                    "markers": clause,
                    "met": all(m in self.index and result.positive[row, self.index[m]] for m in clause),
                    "failing": [m for m in clause if m not in self.index or not result.positive[row, self.index[m]]],
                }
                for clause in groups
            ]
            dropped_for = None
            if result.fired[row, j] and not result.kept[row, j]:
                dropped_for = next(
                    (
                        child
                        for child, parent in self.hierarchy.items()
                        if parent == name and result.fired[row, self.names.index(child)]
                    ),
                    None,
                )
            phenotypes.append(
                {
                    "phenotype": name,
                    "fired": bool(result.fired[row, j]),
                    "dropped_for_child": dropped_for,
                    "criteria": criteria,
                    "strength": float(row_scores[self._own[j]].max()) if self._own[j] else 0.0,
                }
            )
        return {"phenotype": result.phenotypes[cell], "markers": markers, "phenotypes": phenotypes}
