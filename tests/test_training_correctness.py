from types import SimpleNamespace

import pytest
import torch

import image_synthesis.engine.clip_grad_norm as clip_module
import image_synthesis.engine.solver as solver_module
from image_synthesis.engine.clip_grad_norm import ClipGradNorm
from image_synthesis.engine.solver import (
    Solver,
    initialize_pretrained_weights,
    optimizer_step_boundary,
)


@pytest.mark.parametrize(
    "iteration,accumulate,expected",
    [(0, 1, True), (0, 2, False), (1, 2, True), (2, 2, False), (3, 2, True)],
)
def test_optimizer_boundary(iteration, accumulate, expected):
    assert optimizer_step_boundary(
        iteration, step_iteration=1, accumulate_grad_iters=accumulate
    ) is expected


def test_clip_grad_norm_strict_interval(monkeypatch):
    calls = []
    monkeypatch.setattr(clip_module, "clip_grad_norm_", lambda parameters, max_norm: calls.append(max_norm))
    clipper = ClipGradNorm(start_iteration=2, end_iteration=4, max_norm=0.5)
    assert not clipper([], step=1)
    assert clipper([], step=2)
    assert clipper([], step=3)
    assert not clipper([], step=4)
    assert calls == [0.5, 0.5]
    unlimited = ClipGradNorm(start_iteration=2, end_iteration=-1)
    assert unlimited([], step=999)


class _LoadTarget:
    def __init__(self, mismatch=([], [])):
        self.mismatch = mismatch
        self.loaded = []

    def load_state_dict(self, state, strict=False):
        self.loaded.append((state, strict))
        return self.mismatch


class _CheckpointModel(_LoadTarget):
    def __init__(self, mismatch=([], []), ema_mismatch=([], [])):
        super().__init__(mismatch)
        self.transformer = _LoadTarget(ema_mismatch)

    def get_ema_model(self):
        return self.transformer


def test_pretrained_initialization_overlays_live_and_shadow_ema_without_training_state():
    model = _CheckpointModel(mismatch=(["content_codec.vq.external"], []))
    shadow = _LoadTarget()
    state = {
        "model": {"base": 1},
        "ema": {"official_ema": 2},
        "last_iter": 12345,
        "clip_grad_norm": {"last_epoch": 12345},
        "optimizer_and_scheduler": {"must_not_load": True},
    }
    audit = initialize_pretrained_weights(model, shadow, state)
    assert model.loaded == [({"base": 1}, False)]
    assert model.transformer.loaded == [({"official_ema": 2}, False)]
    assert shadow.loaded == [({"official_ema": 2}, True)]
    assert audit["training_state_loaded"] is False


def test_pretrained_initialization_rejects_core_mismatch():
    with pytest.raises(RuntimeError, match="Critical pretrained model"):
        initialize_pretrained_weights(
            _CheckpointModel(mismatch=(["transformer.required"], [])), _LoadTarget(),
            {"model": {}, "ema": {}},
        )
    with pytest.raises(RuntimeError, match="Critical pretrained EMA"):
        initialize_pretrained_weights(
            _CheckpointModel(ema_mismatch=(["required"], [])), _LoadTarget(),
            {"model": {}, "ema": {}},
        )


class _EventLoss:
    def __init__(self, events):
        self.events = events

    def __truediv__(self, value):
        return self

    def backward(self):
        self.events.append("backward")


class _EventModel:
    def __init__(self, events):
        self.events = events

    def __call__(self, **kwargs):
        return {"loss": _EventLoss(self.events)}

    def parameters(self):
        return []


class _Optimizer:
    def __init__(self, events):
        self.events = events

    def step(self):
        self.events.append("optimizer.step")

    def zero_grad(self):
        self.events.append("zero_grad")


class _Scaled:
    def __init__(self, loss):
        self.loss = loss

    def backward(self):
        self.loss.backward()


class _Scaler:
    def __init__(self, events):
        self.events = events

    def scale(self, loss):
        self.events.append("scale")
        return _Scaled(loss)

    def unscale_(self, optimizer):
        self.events.append("unscale")

    def step(self, optimizer):
        self.events.append("scaler.step")

    def update(self):
        self.events.append("scaler.update")


class _EMA:
    def __init__(self, events):
        self.events = events

    def update(self, iteration):
        self.events.append("ema")


def _solver(events, *, amp, accumulation, iteration):
    solver = Solver.__new__(Solver)
    solver.args = SimpleNamespace(amp=amp)
    solver.debug = False
    solver.last_iter = iteration
    solver.last_epoch = 0
    solver.model = _EventModel(events)
    solver.config = {"solver": {"accumulate_grad_iters": accumulation}}
    optimizer = _Optimizer(events)
    solver.optimizer_and_scheduler = {
        "none": {
            "start_iteration": 0, "end_iteration": -1, "start_epoch": 0, "end_epoch": -1,
            "optimizer": {"step_iteration": 1, "module": optimizer},
        }
    }
    solver.clip_grad_norm = lambda parameters, step: events.append("clip")
    solver.ema = _EMA(events)
    if amp:
        solver.scaler = _Scaler(events)
    return solver


def test_amp_unscales_before_clip_and_updates_only_at_boundary(monkeypatch):
    class _Autocast:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    monkeypatch.setattr(solver_module, "autocast", _Autocast)
    events = []
    solver = _solver(events, amp=True, accumulation=2, iteration=0)
    solver.step({}, phase="train")
    assert events == ["scale", "backward"]
    solver.last_iter = 1
    solver.step({}, phase="train")
    assert events[2:] == [
        "scale", "backward", "unscale", "clip", "scaler.step", "zero_grad", "scaler.update", "ema"
    ]


@pytest.mark.parametrize("accumulation", [1, 2])
def test_fp32_clip_step_and_ema_only_on_optimizer_boundary(accumulation):
    events = []
    solver = _solver(events, amp=False, accumulation=accumulation, iteration=0)
    solver.step({}, phase="train")
    if accumulation == 1:
        assert events == ["backward", "clip", "optimizer.step", "zero_grad", "ema"]
    else:
        assert events == ["backward"]


def test_true_resume_restores_full_training_state(monkeypatch, tmp_path):
    state = {
        "last_epoch": 3,
        "last_iter": 99,
        "model": {},
        "ema": {"ema": 1},
        "clip_grad_norm": {"last_epoch": 88},
        "optimizer_and_scheduler": {
            "none": {
                "start_epoch": 7,
                "optimizer": {"step_iteration": 2, "module": {"optimizer": 1}},
            }
        },
    }
    path = tmp_path / "last.pth"
    path.write_bytes(b"stub")
    monkeypatch.setattr(solver_module.torch, "load", lambda *args, **kwargs: state)
    solver = Solver.__new__(Solver)
    solver.args = SimpleNamespace(local_rank=0)
    solver.model = _CheckpointModel()
    solver.ema = _LoadTarget()
    solver.clip_grad_norm = ClipGradNorm()
    optimizer = _LoadTarget()
    solver.optimizer_and_scheduler = {
        "none": {"start_epoch": 0, "optimizer": {"step_iteration": 1, "module": optimizer}}
    }
    solver.logger = SimpleNamespace(log_info=lambda message: None)
    solver.ckpt_dir = str(tmp_path)
    solver.resume(str(path), load_optimizer_and_scheduler=True, load_others=True)
    assert (solver.last_epoch, solver.last_iter) == (3, 99)
    assert solver.clip_grad_norm.last_epoch == 88
    assert solver.ema.loaded == [({"ema": 1}, True)]
    assert optimizer.loaded == [({"optimizer": 1}, False)]
    assert solver.optimizer_and_scheduler["none"]["start_epoch"] == 7
    assert solver.optimizer_and_scheduler["none"]["optimizer"]["step_iteration"] == 2
