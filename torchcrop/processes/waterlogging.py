"""Waterlogging: persistent root-zone anoxia injury and denitrification.

LINTUL-5's only wet-side term is ``RWET``, which throttles the *same day's*
transpiration when the rooted zone loses its air. That misses how waterlogging
actually costs a wheat crop:

* it mostly happens in late winter and spring, when transpiration demand is
  small, so a cut in ``TRAN/PTRAN`` costs almost no assimilate;
* the damage — root death, tiller abortion, lost N — persists for weeks
  after the water has drained, and is largest from stem elongation to
  anthesis;
* anoxic soil denitrifies its nitrate, so a wet spring also starves the crop
  of N.

`Waterlogging` adds those three effects on top of the existing two-zone
bucket — no soil layers. It reads only the rooted-zone moisture the water
balance already tracks; the drainage impedance ``SoilParameters.drain_factor``
is what lets that moisture reach saturation and stay there.

Equations
---------
Aeration index — 0 at the critical air content ``CRAIRC``, 1 at saturation,
the same threshold ``RWET`` uses:

$$
x = \\mathrm{clip}\\!\\left(\\frac{\\theta - (\\theta_\\text{sat} -
\\text{CRAIRC})}{\\text{CRAIRC}},\\ 0,\\ 1\\right)
$$

The injury terms use ``x`` gated by emergence (no crop, no injury);
denitrification is a soil process and uses it ungated.

Injury index ``WLD`` in ``[0, 1]``, accumulated in anoxic degree-days weighted
by the stage sensitivity ``s(DVS)`` (``WLSENSTB``), and recovering on aerated
days:

$$
\\frac{dW}{dt} = k_w\\, x\\, s(\\text{DVS})\\,(T - T_b)_+\\,(1 - W)
- r\\,(1 - x)\\,W
$$

The ``(1 − W)`` factor keeps ``W`` below 1 without a clamp in the rate.
Growth is cut multiplicatively, after the water/nutrient law of the minimum
(an injury, not a resource limitation):

$$
\\text{GTOTAL} \\leftarrow \\text{GTOTAL}\\cdot (1 - W)
$$

Leaf senescence and root extension (both off by default):

$$
\\text{RDR} \\leftarrow \\text{RDR}\\cdot(1 + a_\\text{rdr}\\,W),
\\qquad
\\text{RR} \\leftarrow \\text{RR}\\cdot(1 - a_\\text{root}\\,x)
$$

Denitrification of the soil mineral-N pool (``Q10 = 2`` around 20 °C):

$$
\\text{DENIT} = k_\\text{den}\\, x\\, 2^{(T - 20)/10}\\, \\text{NMINT}
$$

With the defaults ``wl_damage_rate = 0``, ``kdenit = 0``, ``wl_rdr_amp = 0``
and ``wl_root_arrest = 0`` every output is inert and the model is unchanged.

Not modelled here, deliberately: a low-radiation amplification of anoxia
(internal aeration by photosynthetic O₂ matters for aerenchymatous wetland
species, not for wheat) and a grain-number/sink limit from radiation around
flowering. The second is real — it is the photothermal-quotient control of
grain number — but it is a reproductive-sink process, not waterlogging.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from torchcrop.functions.fst_functions import notnul
from torchcrop.functions.interpolation import interpolate
from torchcrop.parameters.crop_params import CropParameters
from torchcrop.parameters.soil_params import SoilParameters
from torchcrop.states.model_state import ModelState


class Waterlogging(nn.Module):
    """Daily waterlogging injury and denitrification rates."""

    def forward(
        self,
        state: ModelState,
        smact: torch.Tensor,
        davtmp: torch.Tensor,
        emerg: torch.Tensor,
        crop_params: CropParameters,
        soil_params: SoilParameters,
    ) -> dict[str, torch.Tensor]:
        """Compute one day of waterlogging rates.

        Args:
            state: Current `ModelState`; uses ``dvs``, ``wl_damage``, ``nmint``.
            smact: Rooted-zone volumetric moisture [m³ m⁻³], shape ``[B]``.
            davtmp: Daily mean temperature [°C], shape ``[B]``.
            emerg: Emergence gate in ``{0, 1}``, shape ``[B]``.
            crop_params: Supplies ``wl_damage_rate``, ``wl_tbase``,
                ``wl_recovery`` and ``wlsenstb``.
            soil_params: Supplies ``wcst``, ``crairc`` and ``kdenit``.

        Returns:
            Dict of ``[B]`` tensors:

            * ``wl_index``       — aeration index ``x`` in ``[0, 1]``
              (ungated by emergence).
            * ``wl_damage_rate`` — ``dW/dt`` [d⁻¹].
            * ``wl_factor``      — growth multiplier ``1 − W``.
            * ``denit``          — mineral-N loss [g N m⁻² d⁻¹].
            * ``rdr_factor``     — leaf death-rate multiplier ``≥ 1``.
            * ``rr_factor``      — root-front advance multiplier in ``[0, 1]``.
        """
        crairc = soil_params.crairc
        smair = soil_params.wcst - crairc
        x_soil = torch.clamp((smact - smair) / notnul(crairc), 0.0, 1.0)
        x = x_soil * emerg

        w = state.wl_damage
        sens = interpolate(crop_params.wlsenstb, state.dvs)
        dd = torch.clamp(davtmp - crop_params.wl_tbase, min=0.0)
        injury = crop_params.wl_damage_rate * x * sens * dd * (1.0 - w)
        recovery = crop_params.wl_recovery * (1.0 - x) * w

        q10 = torch.pow(2.0, (davtmp - 20.0) / 10.0)
        denit = soil_params.kdenit * x_soil * q10 * torch.clamp(state.nmint, min=0.0)

        return {
            "wl_index": x_soil,
            "wl_damage_rate": injury - recovery,
            "wl_factor": 1.0 - w,
            "denit": denit,
            "rdr_factor": 1.0 + crop_params.wl_rdr_amp * w,
            "rr_factor": 1.0 - torch.clamp(crop_params.wl_root_arrest, 0.0, 1.0) * x,
        }
