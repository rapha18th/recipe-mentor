"""
Hard bounds on agent-chosen hyperparameters, per task type.

The LLM's freedom is genuinely in what it proposes for configure_run();
anything outside these ranges is silently clamped, and the clamp is
reported back to the model in the tool's own result -- visible, not
hidden, and the model can see and react to it, same as any other tool
result.
"""
from __future__ import annotations

BOUNDS: dict[str, dict[str, tuple[float, float]]] = {
    "image_classification": {
        "epochs": (3, 25),
        "batch_size": (8, 32),
        "lr": (1e-4, 3e-3),
        #: Real, actionable now (see generic_image_train.py's own
        #: `image_size` param) -- caught live during testing when the
        #: agent proposed a resize-resolution choice via ask_user() that
        #: configure_run had no way to act on yet. AdaptiveAvgPool2d in
        #: SmallCNN makes any size here architecture-compatible.
        "image_size": (32, 224),
    },
    "audio_anomaly": {
        "epochs": (20, 150),
        "batch_size": (8, 32),
        "lr": (1e-4, 3e-3),
    },
}
_INT_PARAMS = {"epochs", "batch_size", "image_size"}


def clamp(task_type: str, epochs: int, batch_size: int, lr: float, **extra: float) -> tuple[dict, dict]:
    """Returns (clamped_values, clamp_notes). clamp_notes is empty when
    nothing needed clamping. `extra` holds task-type-specific knobs (e.g.
    image_size for image_classification) -- silently ignored if the task
    type has no bound for that name, so a stray/irrelevant kwarg from the
    model doesn't raise."""
    bounds = BOUNDS.get(task_type)
    if bounds is None:
        raise ValueError(f"no hyperparameter bounds for task type {task_type!r}")

    notes: dict[str, str] = {}
    out: dict[str, float] = {}
    for name, value in {"epochs": epochs, "batch_size": batch_size, "lr": lr, **extra}.items():
        if name not in bounds:
            continue
        lo, hi = bounds[name]
        clamped = min(max(value, lo), hi)
        if clamped != value:
            notes[name] = f"requested {value}, clamped to {clamped} (allowed range [{lo}, {hi}])"
        out[name] = int(clamped) if name in _INT_PARAMS else clamped
    return out, notes
