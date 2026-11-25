"""Tests for `WetStressOnGrain`, the excess-wetness yield penalty."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from torchcrop.parameters.crop_params import CropParameters
from torchcrop.processes import WetStressOnGrain


def _params(**overrides):
    base = dict(wet_rain_threshold=torch.tensor(1.0), wet_rain_ramp=torch.tensor(0.05),
                wet_begin_devstage=torch.tensor(0.6), wet_end_devstage=torch.tensor(1.3),
                wet_fraction_critical=torch.tensor(0.3), wet_penalty_slope=torch.tensor(1.0),
                wet_penalty_max=torch.tensor(0.5))
    return SimpleNamespace(**(base | overrides))


def _season(wet_every: int, days: int = 100) -> tuple[torch.Tensor, torch.Tensor]:
    dvs = torch.linspace(0.0, 2.0, days).unsqueeze(0)
    rain = torch.zeros(1, days)
    rain[:, ::wet_every] = 10.0
    return rain, dvs


def test_default_crop_is_off():
    rain, dvs = _season(1)
    out = WetStressOnGrain()(rain, dvs, CropParameters())
    assert torch.equal(out["wet_stress_factor"], torch.zeros(1))


def test_factor_is_slope_times_excess_wet_fraction():
    rain, dvs = _season(2)
    out = WetStressOnGrain()(rain, dvs, _params())
    window = out["window_mask"].bool()
    expected = (rain[window] > 1.0).float().mean() - 0.3
    assert out["wet_fraction"].item() == pytest.approx(expected.item() + 0.3, abs=1e-3)
    assert out["wet_stress_factor"].item() == pytest.approx(expected.item(), abs=1e-3)


def test_dry_window_and_cap():
    rain, dvs = _season(1)
    assert WetStressOnGrain()(torch.zeros_like(rain), dvs, _params())["wet_stress_factor"].item() == 0.0
    capped = WetStressOnGrain()(rain, dvs, _params(wet_penalty_slope=torch.tensor(5.0)))
    assert capped["wet_stress_factor"].item() == pytest.approx(0.5)


def test_per_row_parameters_and_gradient():
    rain, dvs = _season(2)
    rain, dvs = rain.repeat(2, 1), dvs.repeat(2, 1)
    slope = torch.tensor([0.0, 1.0], requires_grad=True)
    out = WetStressOnGrain()(rain, dvs, _params(wet_penalty_slope=slope))
    assert out["wet_stress_factor"][0] == 0 and out["wet_stress_factor"][1] > 0
    out["wet_stress_factor"].sum().backward()
    assert slope.grad[1] > 0


def test_model_adjusted_yield_unchanged_when_off():
    from torchcrop.model import Lintul5Model

    weather = torch.rand(2, 30, 8)
    model = Lintul5Model()
    out = model(weather)
    assert torch.equal(out.wet_stress_factor, torch.zeros(2))
