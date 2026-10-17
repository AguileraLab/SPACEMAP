"""The saved settings each Studio run uses, read and changed through each step's own checks.

The assistant sees them (so it can say what a run will do) and can propose changing them before a run; a change
goes through the same checks and files as the step's own Save settings, so classic tabs see it too.
"""

METHODS = (
    "leiden",
    "som",
    "vortex",
    "consensus",
    "subphenotyping",
    "neighbourhoods",
    "interactions",
    "batch_correction",
)


def _editable(method):
    from . import clustering, consensus, correction, interactions, leiden, neighbourhoods, subphenotypes

    return {
        "leiden": lambda: leiden.EDITABLE,
        "som": lambda: clustering._method("som")["editable"],
        "vortex": lambda: clustering._method("vortex")["editable"],
        "consensus": lambda: consensus.EDITABLE,
        "subphenotyping": lambda: subphenotypes.EDITABLE,
        "neighbourhoods": lambda: neighbourhoods.KEYS,
        "interactions": lambda: interactions.KEYS,
        "batch_correction": lambda: correction.EDITABLE,
    }[method]()


def _mode(config):
    """Tab 6's correction mode as the Batch correction step shows it (cyCombine's when none was chosen)."""
    from .correction import _tab6

    chosen = _tab6(config).get("bc_config")
    return (chosen if isinstance(chosen, dict) else {}).get("correction_mode") or "cycombine"


def current(root, method):
    """The settings *method*'s run uses now: {name: value}, only the ones that can be changed."""
    from . import clustering, consensus, correction, interactions, leiden, neighbourhoods, subphenotypes
    from .settings import read_config

    if method not in METHODS:
        raise ValueError("Choose one of " + ", ".join(METHODS) + ".")
    found = {
        "leiden": lambda: leiden.values(root),
        "som": lambda: clustering.values(root, "som"),
        "vortex": lambda: clustering.values(root, "vortex"),
        "consensus": lambda: consensus.values(root),
        "subphenotyping": lambda: subphenotypes.values(root),
        "neighbourhoods": lambda: neighbourhoods.values(read_config(root)),
        "interactions": lambda: interactions.values(read_config(root)),
        "batch_correction": lambda: {
            **correction.config_of(read_config(root)),
            "correction_mode": _mode(read_config(root)),
        },
    }[method]()
    return {key: found.get(key) for key in _editable(method)}


def change(root, method, changes):
    """Save *changes* with the step's own checks; returns (previous values of the changed keys, the new ones)."""
    from . import clustering, consensus, correction, interactions, leiden, neighbourhoods, subphenotypes
    from .settings import read_config, revision

    before = current(root, method)
    if not isinstance(changes, dict) or not changes or set(changes) - set(before):
        raise ValueError(
            f"{method.replace('_', ' ')} has no setting called "
            + ", ".join(sorted(set(changes or {}) - set(before)))
            + "."
        )
    if method == "batch_correction":
        correction.save(root, changes)
    elif method == "neighbourhoods":
        neighbourhoods.save(root, changes, revision(neighbourhoods.values(read_config(root))))
    elif method == "interactions":
        interactions.save(root, changes, revision(interactions.values(read_config(root))))
    elif method in ("som", "vortex"):
        clustering.save(root, method, changes, revision(clustering.values(root, method)))
    else:
        module = {"leiden": leiden, "consensus": consensus, "subphenotyping": subphenotypes}[method]
        module.save(root, changes, revision(module.values(root)))
    after = current(root, method)
    return {key: before[key] for key in changes}, {key: after[key] for key in changes}
