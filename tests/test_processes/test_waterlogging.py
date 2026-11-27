"""Tests for impeded drainage and the `Waterlogging` injury module."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from torchcrop import Lintul5Model
from torchcrop.parameters.crop_params import CropParameters
from torchcrop.parameters.soil_params import SoilParameters
from torchcrop.processes.water_balance import WaterBalance
from torchcrop.processes.waterlogging import Waterlogging
from torchcrop.states.model_state import ModelState
from torchcrop.utils.io import make_constant_weather

WET = dict(rain=10.0, irrad=10.0, davtmp=12.0, tmin=8.0, tmax=16.0)


def _bucket(
    drain_factor: float, days: int = 120, dry_after: int | None = None
) -> tuple[list[dict], SoilParameters]:
    """Integrate the water balance alone under steady heavy rain.

    ``dry_after`` switches the rain off from that day on.
    """
    params = SoilParameters(drain_factor=torch.tensor(drain_factor))
    state = ModelState.initial(batch_size=1, rootdi=0.5, wai=150.0, wa_lower_i=210.0)
    wb = WaterBalance()
    rdm = torch.tensor([1.2])
    out = []
    for t in range(days):
        rain = 0.0 if dry_after is not None and t >= dry_after else 10.0
        day = wb(
            state,
            rain=torch.tensor([rain]),
            pevap=torch.tensor([0.3]),
            ptran=torch.tensor([1.0]),
            params=params,
            rdm=rdm,
        )
        out.append(day)
        state = state.replace(
            wa=state.wa + day["wa_rate"],
            wa_lower=state.wa_lower + day["wa_lower_rate"],
            dslr=torch.clamp(state.dslr + day["dslr_rate"], min=1.0),
            dsos=torch.clamp(state.dsos + day["dsos_rate"], 0.0, 4.0),
        )
    return out, params


def test_free_drainage_never_waterlogs():
    days, params = _bucket(1.0)
    smair = params.wcst - params.crairc
    assert max(float(d["smact"]) for d in days) < float(smair)


def test_impeded_drainage_builds_a_perched_water_table():
    days, params = _bucket(0.05)
    smair = params.wcst - params.crairc
    assert float(days[-1]["smactl"]) > float(params.wcst) - 1e-4
    assert float(days[-1]["smact"]) > float(smair)
    assert max(float(d["smact"]) for d in days) <= float(params.wcst) + 1e-5
    assert max(float(d["smactl"]) for d in days) <= float(params.wcst) + 1e-5
    assert max(abs(float(d["wbal"])) for d in days) < 1e-3
    assert float(days[-1]["runoff"]) > 0.0


def test_injury_needs_anoxia_warmth_and_recovers():
    crop = CropParameters(wl_damage_rate=torch.tensor(0.01))
    soil = SoilParameters()
    wl = Waterlogging()
    state = ModelState.initial(batch_size=3, dvsi=0.5).replace(
        wl_damage=torch.tensor([0.0, 0.0, 0.4])
    )
    saturated = soil.wcst.expand(3).clone()
    smact = torch.stack([saturated[0], saturated[1], soil.wcfc])
    out = wl(
        state,
        smact=smact,
        davtmp=torch.tensor([10.0, -2.0, 10.0]),
        emerg=torch.ones(3),
        crop_params=crop,
        soil_params=soil,
    )
    rate = out["wl_damage_rate"]
    assert torch.isclose(rate[0], torch.tensor(0.01 * 1.0 * 10.0))
    assert rate[1] == 0.0
    assert torch.isclose(rate[2], torch.tensor(-0.02 * 0.4))
    assert torch.allclose(out["wl_factor"], torch.tensor([1.0, 1.0, 0.6]))


def _run(crop: CropParameters, soil: SoilParameters, days: int = 150):
    weather = make_constant_weather(1, days, dtype=torch.float64, **WET)
    model = Lintul5Model(crop_params=crop, soil_params=soil).double()
    return model(weather, start_doy=60)


def test_defaults_are_inert_even_when_waterlogged():
    soil = SoilParameters(drain_factor=torch.tensor(0.05)).to(dtype=torch.float64)
    out = _run(CropParameters().to(dtype=torch.float64), soil)
    assert max(float(d.wl_index) for d in out.diagnostics) > 0.0
    assert all(float(s.wl_damage) == 0.0 for s in out.states)
    assert all(float(d.denit) == 0.0 for d in out.diagnostics)


def test_injury_cuts_growth_with_a_gradient():
    soil = SoilParameters(drain_factor=torch.tensor(0.05)).to(dtype=torch.float64)
    base = _run(CropParameters().to(dtype=torch.float64), soil).biomass[:, -1]

    crop = CropParameters().to(dtype=torch.float64)
    crop.wl_damage_rate = nn.Parameter(torch.tensor(0.002, dtype=torch.float64))
    out = _run(crop, soil)
    assert float(out.states[-1].wl_damage) > 0.0
    assert float(out.biomass[0, -1]) < float(base[0])
    out.biomass[:, -1].sum().backward()
    assert torch.isfinite(crop.wl_damage_rate.grad)
    assert float(crop.wl_damage_rate.grad) < 0.0


def test_drain_factor_has_a_gradient():
    crop = CropParameters(wl_damage_rate=torch.tensor(0.002)).to(dtype=torch.float64)
    soil = SoilParameters().to(dtype=torch.float64)
    soil.drain_factor = nn.Parameter(torch.tensor(0.05, dtype=torch.float64))
    out = _run(crop, soil)
    out.biomass[:, -1].sum().backward()
    assert torch.isfinite(soil.drain_factor.grad)
    assert float(soil.drain_factor.grad) > 0.0


def test_denitrification_depletes_mineral_n():
    soil = SoilParameters(drain_factor=torch.tensor(0.05)).to(dtype=torch.float64)
    base = _run(CropParameters().to(dtype=torch.float64), soil, days=60)
    soil_dn = SoilParameters(
        drain_factor=torch.tensor(0.05), kdenit=torch.tensor(0.05)
    ).to(dtype=torch.float64)
    out = _run(CropParameters().to(dtype=torch.float64), soil_dn, days=60)
    assert max(float(d.denit) for d in out.diagnostics) > 0.0
    assert float(out.states[-1].nmint) < float(base.states[-1].nmint)


def test_perched_water_drains_on_dry_days_at_the_impeded_rate():
    days, params = _bucket(0.05, days=140, dry_after=120)
    kdrain = float(params.ksub * params.drain_factor)
    wet, dry = days[119], days[120:]
    assert float(wet["smactl"]) > float(params.wcst) - 1e-4
    assert all(0.0 < float(d["drain"]) <= kdrain + 1e-6 for d in dry)
    assert float(dry[-1]["smact"]) < float(wet["smact"])
    assert max(abs(float(d["wbal"])) for d in dry) < 1e-3


def test_free_drainage_cascade_is_untouched_by_the_store_term():
    days, _ = _bucket(1.0, days=140, dry_after=120)
    assert all(float(d["drain"]) == 0.0 for d in days[121:])


def test_injury_accelerates_leaf_death_and_anoxia_stops_roots():
    soil = SoilParameters(drain_factor=torch.tensor(0.05)).to(dtype=torch.float64)
    crop = CropParameters(wl_damage_rate=torch.tensor(0.002)).to(dtype=torch.float64)
    base = _run(crop, soil)
    crop_fx = CropParameters(
        wl_damage_rate=torch.tensor(0.002),
        wl_rdr_amp=torch.tensor(2.0),
        wl_root_arrest=torch.tensor(1.0),
    ).to(dtype=torch.float64)
    out = _run(crop_fx, soil)
    assert float(out.states[-1].wlvd) > float(base.states[-1].wlvd)
    assert float(out.states[-1].rootd) < float(base.states[-1].rootd)


def test_fully_rooted_profile_still_drains_its_perched_water():
    """Once roots reach rdm the lower zone is a sliver: its outlet must still pass water."""
    params = SoilParameters(drain_factor=torch.tensor(0.05))
    rdm = torch.tensor([1.2])
    # The lower zone is the 1e-4 m floor, held at field capacity so nothing
    # is spent refilling it.
    state = ModelState.initial(
        batch_size=1, rootdi=1.2, wai=float(1000.0 * params.wcst * 1.2),
        wa_lower_i=float(1000.0 * params.wcfc * 1e-4),
    )
    day = WaterBalance()(
        state, rain=torch.zeros(1), pevap=torch.zeros(1), ptran=torch.zeros(1),
        params=params, rdm=rdm,
    )
    kdrain = float(params.ksub * params.drain_factor)
    assert float(day["drain"]) == pytest.approx(kdrain, rel=1e-4)
    assert float(day["wa_rate"]) == pytest.approx(-kdrain, rel=1e-4)
    assert abs(float(day["wbal"])) < 1e-4
