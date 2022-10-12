from enum import Enum
from functools import partial
from typing import Dict, List, NamedTuple

from conduit.data.structures import SubgroupSample
import matplotlib.pyplot as plt  # type: ignore
import numpy as np
import pandas as pd  # type: ignore
import seaborn as sns  # type: ignore
import torch
from torch import Tensor

from .matching import Direction, MatchedIndicesBD

__all__ = ["MatchingEvaluator"]


class RuleOfThumb(Enum):
    SMD = ("<= 0.1", "0.1, 0.2", ">= 0.2")
    VR = ("<= 4/5", "4/5, 5/4", ">= 5/4")


class MinMax(NamedTuple):
    min: float
    max: float


def _range_rule_of_thumb(x: Tensor, *, bounds: MinMax) -> List[float]:
    min_, max_ = bounds.min, bounds.max
    return [
        torch.count_nonzero(x <= min_).item(),
        torch.count_nonzero((min_ < x) & (x < max_)).item(),
        torch.count_nonzero(x >= max_).item(),
    ]


def _smd_continuous(mu_t: Tensor, *, mu_c: Tensor, sigma_t: Tensor, sigma_c: Tensor) -> Tensor:
    num = mu_t - mu_c
    den = torch.sqrt((sigma_t + sigma_c) / 2)
    return torch.abs(num / den)


def _smd(x_t: Tensor, *, x_c: Tensor) -> Tensor:
    """
    Compute standardised difference in means of two distributions
    TO-DO add reference paper!
    """
    # Take mean, by feature
    mu_t, mu_c = torch.mean(x_t, dim=0), torch.mean(x_c, dim=0)
    sigma_t, sigma_c = torch.var(x_t, dim=0), torch.var(x_c, dim=0)

    return _smd_continuous(mu_t, mu_c=mu_c, sigma_t=sigma_t, sigma_c=sigma_c)


def _vr(x_t: Tensor, *, x_c: Tensor) -> Tensor:
    """
    Compute variance ratio of two distributions
    """
    sigma_t, sigma_c = torch.var(x_t, dim=0), torch.var(x_c, dim=0)
    # inf if den is 0, nan if both num and den are 0s
    return sigma_t / sigma_c


class RuleOfThumbMinMax(Enum):
    SMD = MinMax(min=0.1, max=0.2)
    VR = MinMax(min=4 / 5, max=5 / 4)


class Metric(Enum):
    SMD = partial(_smd)
    VR = partial(_vr)


class MatchingEvaluator:
    @staticmethod
    def compare_counts(
        matched_indices: MatchedIndicesBD[Tensor],
        *,
        data: SubgroupSample[Tensor],
    ) -> pd.DataFrame:
        """
        Number of Units: Pre and Post Matching comparison
        """
        number_units = pd.DataFrame(
            columns=[
                "Treated",
                "Control",
                "Unique Control",
            ],
        )
        # _, number_units.loc["Pre-Matching"] = data.s.unique(return_counts=True) # Wrong order
        _, count = data.s.unique(return_counts=True)
        number_units.loc["Pre-Matching"] = (
            count[_ == 1].item(),
            count[_ == 0].item(),
            count[_ == 0].item(),
        )

        number_units.loc[f"Matched"] = 0
        for direction in Direction:
            match_indices_dir = getattr(matched_indices, str(direction))

            # TODO: Generalise metric to work for k > 1 matches.
            row = [
                len(match_indices_dir),
                len(match_indices_dir),
                len(set(match_indices_dir.match.squeeze(-1).tolist())),
            ]
            number_units.loc[f"Matched - {direction.name.upper()}"] = row
            number_units.loc[f"Matched"] += row
        return number_units.astype(np.float64).reset_index()

    @staticmethod
    def compare_features(
        matched_indices: MatchedIndicesBD[Tensor],
        *,
        data: SubgroupSample[Tensor],
    ) -> Dict[str, pd.DataFrame]:
        rot_df = {
            rule.name: pd.DataFrame(
                index=[
                    "Pre-Matching",
                    "Post-Matching",
                    *[f"Post-Matching - {direction.name.upper()}" for direction in Direction],
                ],
                columns=rule.value,
            )
            for rule in RuleOfThumb
        }

        # Treatment group mask
        mask = data.s == 1
        # Initial Dataset
        for metric_name, metric_f in zip(["SMD", "VR"], [_smd, _vr]):
            metric = metric_f(x_t=data.x[mask], x_c=data.x[~mask])
            metric_rot = _range_rule_of_thumb(metric, bounds=RuleOfThumbMinMax[metric_name].value)
            rot_df[metric_name].loc["Pre-Matching"] = metric_rot

        # Total Matching
        # TODO: Generalise to k > 1 matches.
        x_t_concat_mask = torch.cat(
            (matched_indices.tc.anchor, matched_indices.ct.match.squeeze(-1))
        )
        x_c_concat_mask = torch.cat(
            (matched_indices.tc.match.squeeze(-1), matched_indices.ct.anchor)
        )

        for metric_name, metric_f in zip(["SMD", "VR"], [_smd, _vr]):
            metric = metric_f(x_t=data.x[x_t_concat_mask], x_c=data.x[x_c_concat_mask])
            metric_rot = _range_rule_of_thumb(metric, bounds=RuleOfThumbMinMax[metric_name].value)
            rot_df[metric_name].loc[f"Post-Matching"] = metric_rot

        # Per Direction Matching
        for direction in Direction:
            direction_pairs = getattr(matched_indices, str(direction))
            x_t = data.x[direction_pairs.anchor]
            x_c = data.x[direction_pairs.match.squeeze(-1)]

            if direction is Direction.TC:
                t = "anchor"
                c = "match"
            else:
                c = "match"
                t = "anchor"

            for metric_name, metric_f in zip(["SMD", "VR"], [_smd, _vr]):
                metric = metric_f(x_t=x_t, x_c=x_c)
                metric_rot = _range_rule_of_thumb(
                    metric, bounds=RuleOfThumbMinMax[metric_name].value
                )
                rot_df[metric_name].loc[f"Post-Matching - {direction.name.upper()}"] = metric_rot

        for k, v in rot_df.items():
            rot_df[k] = v.astype(np.float64).reset_index()
        return rot_df
