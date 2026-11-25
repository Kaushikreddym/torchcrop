"""Excess-wetness penalty on grain yield around flowering.

LINTUL-5's only wet-side stress is ``RWET``, the root oxygen shortage of the
water balance, and with a single free-draining layer that never binds outside
standing water. The wet-year losses it therefore misses — a wet, dull
heading-to-grain-fill period, i.e. fungal disease and poor grain set — are
weather-driven, not soil-driven, so this penalty reads rain days, not soil
water.

`WetStressOnGrain` is the trajectory-level sibling of
`torchcrop.processes.heat_stress.HeatStressOnGrain`: it counts wet days over a
development-stage window and converts the wet-day fraction into a
multiplicative yield penalty. The window is in DVS, not calendar months, so one
parameter set transfers across latitudes.

With the default ``wet_penalty_slope = 0`` the factor is exactly zero and the
adjusted yield is unchanged.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from torchcrop.parameters.crop_params import CropParameters


def _column(value: torch.Tensor, like: torch.Tensor) -> torch.Tensor:
    """A scalar or per-row ``[B]`` parameter, broadcastable against ``[B, T]``."""
    value = torch.as_tensor(value, dtype=like.dtype, device=like.device)
    return value.reshape(-1, 1) if value.dim() > 0 else value


class WetStressOnGrain(nn.Module):
    r"""Excess-wetness penalty on grain yield over a DVS window.

    Daily wet indicator, a logistic step at the rain threshold ``P_c`` with
    ramp width ``w`` so ``P_c`` keeps a gradient:

    $$
    d_i = \sigma\!\left(\frac{P_i - P_c}{w}\right)
    $$

    Wet-day fraction over the days with ``DVS`` in
    ``[DVS_begin, DVS_end]``, and the penalty:

    $$
    f = \frac{\sum_i d_i \, m_i}{\sum_i m_i}, \qquad
    \text{WSF} = \mathrm{clip}\big(s\,(f - f_c),\; 0,\; \text{WSF}_{\max}\big)
    $$

    The crop constants are read from `CropParameters`: ``wet_rain_threshold``
    (``P_c``), ``wet_rain_ramp`` (``w``), ``wet_begin_devstage``,
    ``wet_end_devstage``, ``wet_fraction_critical`` (``f_c``),
    ``wet_penalty_slope`` (``s``) and ``wet_penalty_max``. Each may be a scalar
    or one value per row.
    """

    def forward(
        self, rain: torch.Tensor, dvs: torch.Tensor, params: CropParameters
    ) -> dict[str, torch.Tensor]:
        """Compute the wet-stress factor over the flowering window.

        Args:
            rain: Daily precipitation [mm d⁻¹], shape ``[B, T]``.
            dvs: Development-stage trajectory [-], shape ``[B, T]``, aligned
                day-for-day with ``rain`` (post-integration DVS, as fed to
                `HeatStressOnGrain`).
            params: Crop parameters carrying the ``wet_*`` fields.

        Returns:
            Dict of tensors:

            * ``wet_stress_factor`` [-] — ``WSF`` in ``[0, WSF_max]``,
              shape ``[B]``; ``0`` when the window never opens.
            * ``wet_fraction`` [-] — ``f``, shape ``[B]``.
            * ``daily_wet`` [-] — ``d_i``, shape ``[B, T]``.
            * ``window_mask`` [-] — ``m_i``, shape ``[B, T]``.
        """
        col = lambda name: _column(getattr(params, name), rain)  # noqa: E731
        daily_wet = torch.sigmoid((rain - col("wet_rain_threshold")) / col("wet_rain_ramp"))
        window_mask = (
            (dvs >= col("wet_begin_devstage")) & (dvs <= col("wet_end_devstage"))
        ).to(rain.dtype)
        window_days = window_mask.sum(dim=1)
        wet_fraction = (daily_wet * window_mask).sum(dim=1) / window_days.clamp(min=1.0)
        flat = lambda name: _column(getattr(params, name), rain).reshape(-1)  # noqa: E731
        raw = flat("wet_penalty_slope") * (wet_fraction - flat("wet_fraction_critical"))
        # Clamp against the per-row maximum with min/max, not torch.clamp: the
        # bound is a tensor that may differ per row.
        factor = torch.minimum(raw.clamp(min=0.0), flat("wet_penalty_max"))
        factor = factor * (window_days > 0).to(rain.dtype)
        return {
            "wet_stress_factor": factor,
            "wet_fraction": wet_fraction,
            "daily_wet": daily_wet,
            "window_mask": window_mask,
        }
