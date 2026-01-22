from abc import ABC, abstractmethod
from typing import Any, Callable, List, Literal, Optional, Union

import torch
from torch import Tensor, tensor
from torchmetrics import Metric
import math

from torchmetrics.utilities.checks import _check_retrieval_inputs
from torchmetrics.utilities.data import _flexible_bincount, dim_zero_cat



def _retrieval_aggregate(
    values: Tensor,
    aggregation: Union[Literal["mean", "median", "min", "max"], Callable] = "mean",
    dim: Optional[int] = None,
) -> Tensor:
    """Aggregate the final retrieval values into a single value."""
    if aggregation == "mean":
        return values.mean() if dim is None else values.mean(dim=dim)
    if aggregation == "median":
        return values.median() if dim is None else values.median(dim=dim).values
    if aggregation == "min":
        return values.min() if dim is None else values.min(dim=dim).values
    if aggregation == "max":
        return values.max() if dim is None else values.max(dim=dim).values
    return aggregation(values, dim=dim)

class CustomRetrievalMetric(Metric, ABC):
    """Works with binary target data. Accepts float predictions from a model output.

    As input to ``forward`` and ``update`` the metric accepts the following input:

    - ``preds`` (:class:`~torch.Tensor`): A float tensor of shape ``(N, ...)``
    - ``target`` (:class:`~torch.Tensor`): A long or bool tensor of shape ``(N, ...)``
    - ``indexes`` (:class:`~torch.Tensor`): A long tensor of shape ``(N, ...)`` which indicate to which query a
      prediction belongs

    .. hint::
        The ``indexes``, ``preds`` and ``target`` must have the same dimension and will be flattened
        to single dimension once provided.

    .. attention::
        Predictions will be first grouped by ``indexes`` and then the real metric, defined by overriding
        the `_metric` method, will be computed as the mean of the scores over each query.

    As output to ``forward`` and ``compute`` the metric returns the following output:

    - ``metric`` (:class:`~torch.Tensor`): A tensor as computed by ``_metric`` if the number of positive targets is
      at least 1, otherwise behave as specified by ``self.empty_target_action``.

    Args:
        empty_target_action:
            Specify what to do with queries that do not have at least a positive
            or negative (depend on metric) target. Choose from:

            - ``'neg'``: those queries count as ``0.0`` (default)
            - ``'pos'``: those queries count as ``1.0``
            - ``'skip'``: skip those queries; if all queries are skipped, ``0.0`` is returned
            - ``'error'``: raise a ``ValueError``

        ignore_index:
            Ignore predictions where the target is equal to this number.
        aggregation:
            Specify how to aggregate over indexes. Can either a custom callable function that takes in a single tensor
            and returns a scalar value or one of the following strings:

            - ``'mean'``: average value is returned
            - ``'median'``: median value is returned
            - ``'max'``: max value is returned
            - ``'min'``: min value is returned

        kwargs: Additional keyword arguments, see :ref:`Metric kwargs` for more info.

    Raises:
        ValueError:
            If ``empty_target_action`` is not one of ``error``, ``skip``, ``neg`` or ``pos``.
        ValueError:
            If ``ignore_index`` is not `None` or an integer.

    """

    is_differentiable: bool = False
    higher_is_better: bool = True
    full_state_update: bool = False

    indexes: List[Tensor]
    preds: List[Tensor]
    target: List[Tensor]

    def __init__(
        self,
        empty_target_action: str = "neg",
        ignore_index: Optional[int] = None,
        aggregation: Union[Literal["mean", "median", "min", "max"], Callable] = "mean",
        top_k: int = 10,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.allow_non_binary_target = False

        empty_target_action_options = ("error", "skip", "neg", "pos")
        if empty_target_action not in empty_target_action_options:
            raise ValueError(f"Argument `empty_target_action` received a wrong value `{empty_target_action}`.")
        self.empty_target_action = empty_target_action

        if ignore_index is not None and not isinstance(ignore_index, int):
            raise ValueError("Argument `ignore_index` must be an integer or None.")
        self.ignore_index = ignore_index

        if not (aggregation in ("mean", "median", "min", "max") or callable(aggregation)):
            raise ValueError(
                "Argument `aggregation` must be one of `mean`, `median`, `min`, `max` or a custom callable function"
                f"which takes tensor of values, but got {aggregation}."
            )
        self.aggregation = aggregation

        self.add_state("indexes", default=[], dist_reduce_fx=None)
        self.add_state("preds", default=[], dist_reduce_fx=None)
        self.add_state("target", default=[], dist_reduce_fx=None)
        self.k = top_k

    def update(self, preds: Tensor, target: Tensor, indexes: Tensor) -> None:
        """Check shape, check and convert dtypes, flatten and add to accumulators."""
        if indexes is None:
            raise ValueError("Argument `indexes` cannot be None")

        self.indexes.append(indexes)
        self.preds.append(preds)
        self.target.append(target)

    def compute(self) -> Tensor:
        """First concat state ``indexes``, ``preds`` and ``target`` since they were stored as lists.

        After that, compute list of groups that will help in keeping together predictions about the same query. Finally,
        for each group compute the ``_metric`` if the number of positive targets is at least 1, otherwise behave as
        specified by ``self.empty_target_action``.

        """
        res = []
        for mini_preds, mini_target in zip(
            self.preds, 
            self.target
        ):
            res.append(self._metric(mini_preds, mini_target))
        return _retrieval_aggregate(torch.cat([x.to(self.preds[0]) for x in res]), self.aggregation)

    @abstractmethod
    def _metric(self, preds: Tensor, target: Tensor) -> Tensor:
        """Compute a metric over a predictions and target of a single group.

        This method should be overridden by subclasses.

        """

class RetrievalPrecision(CustomRetrievalMetric):
    def _metric(self, preds: Tensor, target: Tensor) -> Tensor:
        """
        Computes Recall@K for a batch of predictions.
        
        Args:
            scores (torch.Tensor): Predicted scores, shape (bs, N)
            ground_truth (torch.Tensor): Ground truth labels, shape (bs, N)
            k (int): The cutoff rank K

        Returns:
            float: Average Recall@K across the batch
        """
        # Get the indices of the top-K items for each sample
        bs, N = preds.shape
        # Get top-k indices by predicted score
        topk_indices = preds.topk(self.k, dim=1, largest=True).indices

        # Gather relevant labels at the top-k positions
        topk_targets = torch.gather(target, dim=1, index=topk_indices)

        # Number of relevant items in the top-k
        relevant_at_k = topk_targets.sum(dim=1)

        # Precision = (# relevant in top-k) / k
        precision = relevant_at_k / self.k

        return precision
    
class RetrievalMRR(CustomRetrievalMetric):
    def _metric(self, preds: Tensor, target: Tensor) -> Tensor:
        """
        Computes Mean Reciprocal Rank (MRR) over a batch.
        """
        bs, N = preds.shape
        # Sort indices by descending score
        sorted_indices = preds.argsort(dim=1, descending=True)

        # Gather ground truth in sorted order
        sorted_targets = torch.gather(target, dim=1, index=sorted_indices)

        # Find the rank of the first relevant item
        reciprocal_ranks = torch.zeros(bs, device=preds.device)
        for i in range(bs):
            relevant = (sorted_targets[i] == 1).nonzero(as_tuple=False)
            if relevant.numel() > 0:
                reciprocal_ranks[i] = 1.0 / (relevant[0].item() + 1)

        return reciprocal_ranks

class RetrievalRecall(CustomRetrievalMetric):
    def _metric(self, preds: Tensor, target: Tensor) -> Tensor:
        """
        Computes global Recall: TP / (TP + FN) over all elements.
        """
        bs, N = preds.shape
        # Get top-k indices by predicted score
        topk_indices = preds.topk(self.k, dim=1, largest=True).indices

        # Gather relevant targets at top-k positions
        topk_targets = torch.gather(target, dim=1, index=topk_indices)

        # Number of relevant items in the top-k
        relevant_at_k = topk_targets.sum(dim=1)

        # Total number of relevant items per instance
        total_relevant = target.sum(dim=1).clamp(min=1)  # avoid division by zero

        # Recall = (# relevant in top-k) / (# total relevant)
        recall = relevant_at_k / total_relevant

        return recall
    
class RetrievalNormalizedDCG(CustomRetrievalMetric):
    def __init__(self, top_k: int, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.k = top_k

    def _dcg(self, relevances: Tensor) -> Tensor:
        """
        Computes Discounted Cumulative Gain.
        """
        discounts = torch.log2(torch.arange(2, 2 + relevances.size(1), device=relevances.device).float())
        return (relevances / discounts).sum(dim=1)

    def _metric(self, preds: Tensor, target: Tensor) -> Tensor:
        """
        Computes nDCG@K
        """
        # Get top-K indices
        topk_indices = torch.topk(preds, k=self.k, dim=1).indices
        # Relevance of top-K predictions
        topk_relevances = torch.gather(target, dim=1, index=topk_indices)

        # DCG for predicted ranking
        dcg = self._dcg(topk_relevances)

        # Ideal DCG (sort target per row)
        ideal_relevances, _ = torch.topk(target, self.k, dim=1)
        ideal_dcg = self._dcg(ideal_relevances).clamp(min=1e-8)  # avoid divide by 0

        ndcg = dcg / ideal_dcg
        return ndcg