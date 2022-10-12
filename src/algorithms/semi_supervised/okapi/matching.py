from dataclasses import replace
from enum import Enum
from typing import Dict, Optional, Type, TypeVar, Union, cast, overload

import attr
from conduit.data.structures import InputContainer, SubgroupSample, concatenate_inputs
import numpy as np
import numpy.typing as npt
from ranzen.misc import gcopy
from sklearn.linear_model import LogisticRegression  # type: ignore
from sklearn.preprocessing import StandardScaler  # type: ignore
import torch
from torch import Tensor
import torch.nn.functional as F
import torch_scatter  # type: ignore
from typing_extensions import Self

from src.utils import to_numpy

__all__ = [
    "BinaryCaliperNN",
    "CaliperNN",
    "Direction",
    "LrSolver",
    "MatchedIndices",
    "MatchedIndicesBD",
]


class Direction(Enum):
    TC = 1
    CT = 0

    def __str__(self) -> str:
        return self.name.lower()


I = TypeVar("I", Tensor, npt.NDArray[np.int64])


@attr.define(kw_only=True, eq=False)
class MatchedIndices(InputContainer[I]):
    anchor: I
    match: I

    def __attrs_post_init__(self) -> None:
        if len(self.anchor) != len(self.match):
            raise AttributeError(
                "Number of 'anchor' indices should match the number of 'match' indices."
            )

    def __len__(self) -> int:
        return len(self.anchor)

    def __add__(self, other: Self) -> Self:
        copy = gcopy(self, deep=False)
        copy.anchor = concatenate_inputs(x1=copy.anchor, x2=other.anchor, is_batched=True)
        copy.match = concatenate_inputs(x1=copy.match, x2=other.match, is_batched=True)
        return copy

    def numpy(self) -> "MatchedIndices[npt.NDArray[np.int64]]":
        obj = self
        if isinstance(self.anchor, Tensor):
            match = cast(Tensor, self.match)
            match = to_numpy(match).astype(np.int64)
            anchor = to_numpy(self.anchor).astype(np.int64)
            return MatchedIndices(anchor=anchor, match=match)
        return cast("MatchedIndices[npt.NDArray[np.int64]]", obj)

    def __getitem__(self: "MatchedIndices[Tensor]", index: Tensor) -> "MatchedIndices[Tensor]":
        return gcopy(self, anchor=self.anchor[index], match=self.match[index], deep=False)

    @classmethod
    def from_dict(cls: Type[Self], dict_: Dict[str, I]) -> "MatchedIndices[I]":
        return cls(**dict_)


@attr.define(kw_only=True, eq=False)
class MatchedIndicesBD(InputContainer[I]):
    tc: MatchedIndices[I]
    ct: MatchedIndices[I]

    @classmethod
    def from_dict(cls: Type[Self], dict_: Dict[str, Dict[str, I]]) -> "MatchedIndicesBD[I]":
        tc = MatchedIndices(**dict_["tc"])
        ct = MatchedIndices(**dict_["ct"])
        return cls(tc=tc, ct=ct)

    @property
    def anchor(self) -> I:
        return concatenate_inputs(
            x1=self.tc.anchor,
            x2=self.ct.anchor,
            is_batched=True,
        )

    @property
    def match(self) -> I:
        return concatenate_inputs(
            x1=self.tc.match,
            x2=self.ct.match,
            is_batched=True,
        )

    def __len__(self) -> int:
        return len(self.tc)

    def __add__(self, other: Self) -> Self:
        copy = gcopy(self, deep=False)
        copy.tc = copy.tc + other.tc
        copy.ct = copy.tc + other.ct
        return copy

    def numpy(self) -> "MatchedIndicesBD[npt.NDArray[np.int64]]":
        obj = self
        if isinstance(self.tc.anchor, Tensor):
            ct = cast("MatchedIndices[Tensor]", self.ct)
            obj = MatchedIndicesBD(tc=self.tc.numpy(), ct=ct.numpy())
        return cast("MatchedIndicesBD[npt.NDArray[np.int64]]", obj)

    def __getitem__(self: "MatchedIndicesBD[Tensor]", index: Tensor) -> "MatchedIndicesBD[Tensor]":
        return gcopy(self, tc=self.tc[index], ct=self.ct[index], deep=False)


class LrSolver(Enum):
    NEWTON_CG = "newton-cg"
    LBFGS = "lbfgs"
    LIBLINEAR = "liblinear"
    SAG = "sag"
    SAGA = "saga"


@attr.define(kw_only=True, eq=False)
class CaliperNNBase:
    fixed_caliper_max: float = 0.9
    std_caliper: Optional[float] = 0.2
    twoway_caliper: bool = True
    temperature: float = 1.0
    reweight: bool = True
    c: float = 1
    normalize: bool = False
    solver: LrSolver = LrSolver.LBFGS
    train_frac: float = 1.0
    k: int = 1

    @torch.no_grad()
    def _temper_probs(self, prob_pos: Tensor) -> Tensor:
        logits_pos = torch.logit(prob_pos)
        return (logits_pos / self.temperature).sigmoid()

    @torch.no_grad()
    def _compute_propensity_scores(self, inputs: SubgroupSample[Tensor]) -> Tensor:
        """
        Generate the propensity scores using a logistic regression model.
        """
        # Initialise propensity scorer.
        clf = LogisticRegression(
            solver=self.solver.value,
            random_state=1,
            tol=1e-12,
            max_iter=1000,
            C=self.c,
            multi_class="ovr",
        )
        x = inputs.x
        s = inputs.s
        # Sample a subset of the data if train_frac < 1, else proceed with the full dataset.
        if self.train_frac < 1:
            n_total = len(inputs.x)
            n_train = round(self.train_frac * n_total)
            train_idxs = torch.randperm(len(inputs.x))[:n_train]
            x = x[train_idxs]
            s = s[train_idxs]

        scaler = StandardScaler()
        x_np = scaler.fit_transform(to_numpy(x))
        s_np = to_numpy(s)

        sample_weights = None
        # Compute weights based on the complement of the subgroup-counts.
        if self.reweight:
            _, inverse, counts = s.unique(return_counts=True, return_inverse=True)
            counts_r = counts.reciprocal()
            weights = counts_r / counts_r.sum()
            sample_weights = to_numpy(weights[inverse])

        clf.fit(X=x_np, y=s_np, sample_weight=sample_weights)

        prob_clf = clf.predict_proba(x_np)
        # If the task is a binary one take just the first element of the
        # predicted-probability vectors.
        if prob_clf.shape[1] == 2:
            prob_clf = prob_clf[:, 0]
        propensity_scores = torch.as_tensor(prob_clf, device=inputs.x.device)
        # Temperature scaling
        if self.temperature != 1:
            propensity_scores = self._temper_probs(propensity_scores)

        return propensity_scores


@attr.define(kw_only=True, eq=False)
class BinaryCaliperNN(CaliperNNBase):
    @property
    def fixed_caliper_min(self) -> float:
        return 1 - self.fixed_caliper_max

    @torch.no_grad()
    def _match(
        self,
        queries: SubgroupSample,
        keys: Optional[SubgroupSample] = None,
        *,
        direction: Direction,
        ps_query: Optional[Tensor] = None,
        ps_key: Optional[Tensor] = None,
    ) -> MatchedIndices[Tensor]:

        if ps_query is None:
            if keys is None:
                ps_query = self._compute_propensity_scores(inputs=queries)
            else:
                ps_query_key = self._compute_propensity_scores(inputs=queries + keys)
                ps_query, ps_key_t = ps_query_key.tensor_split([len(queries.x)], dim=0)
                if ps_key is None:
                    ps_key = ps_key_t
        ps_query = ps_query.view(len(queries.x))

        if no_keys := (keys is None):
            keys = queries
            ps_key = ps_query
        else:
            if ps_key is None:
                ps_key = self._compute_propensity_scores(inputs=queries + keys)[len(queries.x) :]
            ps_key = ps_key.view(len(keys.x))

        # Mask selecting 'anchor' samples.
        anchor_mask = queries.s == int(direction.value)
        # Mask selecting 'candidate-match' samples.
        match_mask = keys.s != int(direction.value)

        # Apply the 'fixed caliper' if required
        fixed_caliper_mask = (ps_query > (self.fixed_caliper_min)) & (
            ps_query < self.fixed_caliper_max
        )
        # PS and Index of  anchor and match-candidates after applying the fixed caliper
        calipered_anchor_mask = fixed_caliper_mask & anchor_mask

        if self.normalize:
            queries = replace(queries, x=F.normalize(queries.x, dim=1, p=2))
            keys = replace(keys, x=F.normalize(keys.x, dim=1, p=2))

        x_anchor = queries.x[calipered_anchor_mask]
        ps_anchor = ps_query[calipered_anchor_mask]

        # Apply the 'fixed caliper' if required
        fixed_caliper_mask = (ps_key > self.fixed_caliper_min) & (ps_key < self.fixed_caliper_max)
        calipered_match_mask = (
            (fixed_caliper_mask & match_mask) if self.twoway_caliper else match_mask
        )

        x_match = keys.x[calipered_match_mask]
        ps_match = ps_key[calipered_match_mask]

        # Compute Propensity Score standard deviation; needed for the std-distance caliper
        var_anchor_ps = torch.var(ps_anchor)
        var_ps_match = torch.var(ps_match)
        std_ps = torch.sqrt(0.5 * (var_anchor_ps + var_ps_match))

        # Compute Euclidean distances for each pair.
        pw_dists_ps = torch.cdist(x1=ps_anchor.view(-1, 1), x2=ps_match.view(-1, 1), p=2)

        # If the std_caliper is defined (not None), use it to define a threshold on the propensity
        # score distance
        if self.std_caliper is not None:
            std_caliper_t = self.std_caliper * std_ps
        # Otherwise, do not eliminate possible anchors based on it; N.B. all the distances
        # are <= than the overall max distance
        else:
            std_caliper_t = torch.max(torch.max(pw_dists_ps, dim=0).values)

        # Compute the pairwise distances of the samples
        pw_dists_x = torch.cdist(x1=x_anchor, x2=x_match, p=2)
        # Set pairwise distances betweeen to inf if the propensity score distance is above
        # the pre-defined threshold
        sc_mask = pw_dists_ps > std_caliper_t
        pw_dists_x[sc_mask] = float("inf")
        # nbr_dists, nbr_inds = torch.min(pw_dists_x, dim=1)
        nbr_dists, nbr_inds = torch.topk(pw_dists_x, dim=1, largest=False, k=self.k)
        is_matched = ~nbr_dists.isinf().any(dim=1)

        anchor_inds = (calipered_anchor_mask.nonzero()[is_matched]).flatten()
        match_inds = (calipered_match_mask.nonzero()[nbr_inds[is_matched]]).view(-1, self.k)

        return MatchedIndices(anchor=anchor_inds, match=match_inds)

    @overload
    def __call__(
        self,
        queries: SubgroupSample,
        *,
        keys: Optional[SubgroupSample] = ...,
        direction: None,
        ps_query: Optional[Tensor] = ...,
        ps_key: Optional[Tensor] = ...,
    ) -> MatchedIndicesBD[Tensor]:
        ...

    @overload
    def __call__(
        self,
        queries: SubgroupSample,
        *,
        keys: Optional[SubgroupSample] = ...,
        direction: Direction = ...,
        ps_query: Optional[Tensor] = ...,
        ps_key: Optional[Tensor] = ...,
    ) -> MatchedIndices[Tensor]:
        ...

    @torch.no_grad()
    def __call__(
        self,
        queries: SubgroupSample,
        *,
        keys: Optional[SubgroupSample] = None,
        direction: Optional[Direction] = Direction.TC,
        ps_query: Optional[Tensor] = None,
        ps_key: Optional[Tensor] = None,
    ) -> Union[MatchedIndices[Tensor], MatchedIndicesBD[Tensor]]:
        if direction is None:
            return MatchedIndicesBD(
                **{
                    str(direction): self._match(
                        queries=queries,
                        keys=keys,
                        direction=direction,
                        ps_query=ps_query,
                        ps_key=ps_key,
                    )
                    for direction in Direction
                }
            )

        return self._match(
            queries=queries,
            keys=keys,
            direction=direction,
            ps_query=ps_query,
            ps_key=ps_key,
        )


@attr.define(kw_only=True, eq=False)
class MeanVar:
    mean: Tensor
    var: Tensor


def groupwise_mean_var(input: Tensor, *, index: Tensor, dim: int = 0) -> MeanVar:
    output_size = input.size(1)
    gw_mean = torch_scatter.scatter(
        input, dim=dim, index=index, reduce="mean", dim_size=output_size
    )
    sq_diff = (input.squeeze(-1) - gw_mean[index]).pow(2)
    gw_var = torch_scatter.scatter(sq_diff, index=index, dim=0, reduce="mean", dim_size=output_size)
    return MeanVar(mean=gw_mean, var=gw_var)


@attr.define(kw_only=True, eq=False)
class CaliperNN(CaliperNNBase):
    @torch.no_grad()
    def _match(
        self,
        queries: SubgroupSample,
        *,
        keys: Optional[SubgroupSample] = None,
        ps_query: Optional[Tensor] = None,
        ps_key: Optional[Tensor] = None,
    ) -> MatchedIndices[Tensor]:
        if ps_query is None:
            if keys is None:
                ps_query = self._compute_propensity_scores(inputs=queries)
            else:
                ps_query_key = self._compute_propensity_scores(inputs=queries + keys)
                ps_query, ps_key_t = ps_query_key.tensor_split([len(queries.x)], dim=0)
                if ps_key is None:
                    ps_key = ps_key_t
        ps_query = ps_query.view(len(queries.x), -1)
        if ps_query.size(1) == 1:
            # We assume that '1' corresponds to the 'positive' class.
            ps_query = torch.cat((1 - ps_query, ps_query), dim=-1)

        if no_keys := (keys is None):
            keys = queries
            ps_key = ps_query
        else:
            if ps_key is None:
                ps_key = self._compute_propensity_scores(inputs=queries + keys)[len(queries.x) :]
            ps_key = ps_key.view(len(keys.x), -1)
            if ps_key.size(1) == 1:
                # We assume that '1' corresponds to the 'positive' class.
                ps_key = torch.cat((1 - ps_key, ps_key), dim=-1)

        # Only prermit samples to be matched with samples belonging to other groups.
        diff_group_mask = queries.s.view(-1, 1) != keys.s.flatten()
        # Filter out any samples which do not have any dimensions that satisfy the fixed-caliper
        # threshold.
        fc_filter = ps_query.max(dim=1).values < self.fixed_caliper_max
        query_filter = fc_filter

        if self.normalize:
            queries = replace(queries, x=F.normalize(queries.x, dim=1, p=2))
            keys = replace(keys, x=F.normalize(keys.x, dim=1, p=2))

        if self.twoway_caliper:
            # The queries and keys are one and the same so there's no need to compute a separate
            # caliper-based mask.
            if no_keys:
                key_filter = fc_filter
            else:
                # Compute the caliper-based mask for the keys.
                key_filter = ps_key.max(dim=1).values < self.fixed_caliper_max
            x_key = keys.x[key_filter]
            s_key = keys.s[key_filter]
            ps_key = ps_key[key_filter]
            diff_group_mask = diff_group_mask[:, key_filter]
        else:
            key_filter = slice(None)
            x_key = keys.x
            s_key = keys.s
            ps_key = ps_key

        # Filter out any samples which do not have any potential cross-group matches.
        query_filter &= diff_group_mask.any(dim=1)
        x_query = queries.x[query_filter]
        # Only proceed with matching if there are any queries remaining after filtering.
        if len(x_query) == 0:
            anchor_inds = x_query.new_tensor([], dtype=torch.long)
            match_inds = anchor_inds.clone()
        else:
            ps_query = ps_query[query_filter]
            # Create a mask indicating which samples (both query and key) should be excluded from
            # matching .
            exclusion_mask = ~diff_group_mask[query_filter]

            # Compute the pairwise distances of the propensity scores..
            pw_dists_ps = (ps_query.unsqueeze(1) - ps_key.unsqueeze(0)).pow(2).sqrt()
            s_key_t = s_key.view(1, -1, 1).expand(pw_dists_ps.size(0), -1, -1)
            pw_dists_ps = pw_dists_ps.gather(-1, s_key_t).squeeze(-1)

            # Compute the groupwise stds of the propensity scores for the std-caliper.
            if no_keys:
                index = queries.s[query_filter]
                var_ps = groupwise_mean_var(ps_query, index=index, dim=0).var
            else:
                index = torch.cat((queries.s[query_filter], keys.s[key_filter]), dim=0)
                var_ps = groupwise_mean_var(torch.cat((ps_query, ps_key), dim=0), index=index).var
            std_ps = var_ps.mean(dim=0).sqrt().gather(dim=0, index=s_key)
            # If the std_caliper is defined, use it to define a threshold on the propensity
            # score distance.
            if self.std_caliper is not None:
                std_caliper_t = self.std_caliper * std_ps
            else:
                std_caliper_t = torch.max(torch.max(pw_dists_ps, dim=0).values)

            # Compute Euclidean distances for each cross-group pair of samples.
            pw_dists_x = torch.cdist(x1=x_query, x2=x_key, p=2)
            sc_mask = pw_dists_ps > std_caliper_t
            # Update the exclusion matrix using the std-caliper
            exclusion_mask |= sc_mask
            pw_dists_x[exclusion_mask] = float("inf")
            nbr_dists, nbr_inds = torch.topk(pw_dists_x, dim=1, largest=False, k=self.k)
            is_matched = ~nbr_dists.isinf().any(dim=1)

            query_inds = query_filter.nonzero().squeeze(-1)
            anchor_inds = (query_inds[is_matched]).flatten()
            if self.twoway_caliper:
                if isinstance(key_filter, Tensor):
                    row_inds_m = key_filter.nonzero().squeeze(-1)
                    match_inds = row_inds_m[nbr_inds[is_matched]]
                else:
                    match_inds = nbr_inds[is_matched]
            else:
                match_inds = nbr_inds[is_matched]
            match_inds = match_inds.view(-1, self.k)

        return MatchedIndices(anchor=anchor_inds, match=match_inds)

    @torch.no_grad()
    def __call__(
        self,
        queries: SubgroupSample,
        *,
        keys: Optional[SubgroupSample] = None,
        ps_query: Optional[Tensor] = None,
        ps_key: Optional[Tensor] = None,
    ) -> MatchedIndices[Tensor]:
        return self._match(
            queries=queries,
            keys=keys,
            ps_query=ps_query,
            ps_key=ps_key,
        )
