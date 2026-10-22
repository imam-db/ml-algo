"""
Association Rule Mining from Scratch
====================================

This module implements the two classic frequent-itemset mining algorithms and
the standard association-rule metrics using only NumPy and pandas:

- transaction encoding helpers (list of baskets -> one-hot matrix),
- ``support``, ``confidence``, ``lift``, ``leverage`` and ``conviction``,
- **Apriori** with a level-wise join step and anti-monotone subset pruning,
- **FP-Growth** with an FP-tree, header table and conditional pattern bases,
- rule generation filtered by minimum confidence,
- ``generate_warung_transactions`` for reproducible synthetic Indonesian
  warung/minimarket baskets with planted product associations.

The demo at the bottom mines rules from a synthetic warung dataset.

Author: ML-Algo Repository
"""

from __future__ import annotations

from collections import Counter, defaultdict
from itertools import combinations
from math import inf
from typing import Dict, FrozenSet, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd


Itemset = FrozenSet[str]
TransactionInput = Union[Iterable[Iterable[str]], pd.DataFrame]

RULES_COLUMNS = [
    "antecedents",
    "consequents",
    "antecedent_support",
    "consequent_support",
    "support",
    "confidence",
    "lift",
    "leverage",
    "conviction",
]


# ---------------------------------------------------------------------------
# Transaction encoding
# ---------------------------------------------------------------------------
def encode_transactions(transactions: TransactionInput) -> List[FrozenSet[str]]:
    """
    Normalize baskets into a list of frozensets.

    Parameters
    ----------
    transactions : iterable of iterables, or pandas.DataFrame
        Either a sequence of baskets (for example ``[["Indomie", "Telur"], ...]``)
        or a one-hot/boolean DataFrame whose columns are items. A single string
        is treated as one one-item basket.

    Returns
    -------
    list of frozenset of str
        One frozenset per transaction. Duplicate items inside a basket are
        removed; empty baskets are preserved so that support denominators stay
        correct.

    Notes
    -----
    Non-string items are converted to strings so that item ordering and joins
    are deterministic. A DataFrame is treated as a boolean matrix: a cell is
    truthy when the item is present in that transaction.
    """
    encoded: List[FrozenSet[str]] = []

    if isinstance(transactions, pd.DataFrame):
        columns = [str(col) for col in transactions.columns]
        values = transactions.to_numpy()
        for row in values:
            items = {columns[j] for j, value in enumerate(row) if bool(value)}
            encoded.append(frozenset(items))
        return encoded

    if isinstance(transactions, (str, bytes)):
        transactions = [transactions]

    for transaction in transactions:
        if isinstance(transaction, (str, bytes)):
            raw_items = {transaction}
        else:
            try:
                raw_items = set(transaction)
            except TypeError as exc:
                raise ValueError(
                    "Each transaction must be an iterable of items"
                ) from exc

        normalized = set()
        for item in raw_items:
            if item is None:
                raise ValueError("Transactions must not contain None items")
            normalized.add(item if isinstance(item, str) else str(item))
        encoded.append(frozenset(normalized))

    return encoded


def one_hot_encode(
    transactions: TransactionInput,
    item_order: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """
    Encode baskets as a boolean item matrix (one column per item).

    Parameters
    ----------
    transactions : iterable of iterables, or pandas.DataFrame
        Baskets to encode.
    item_order : sequence of str, optional
        Explicit column order. By default the items are sorted.

    Returns
    -------
    pandas.DataFrame
        Boolean DataFrame with one row per transaction and one column per item.
    """
    encoded = encode_transactions(transactions)
    if item_order is None:
        items = sorted({item for basket in encoded for item in basket})
    else:
        items = [str(item) for item in item_order]

    positions = {item: j for j, item in enumerate(items)}
    matrix = np.zeros((len(encoded), len(items)), dtype=bool)
    for i, basket in enumerate(encoded):
        for item in basket:
            j = positions.get(item)
            if j is not None:
                matrix[i, j] = True

    return pd.DataFrame(
        matrix,
        columns=pd.Index(items),
        index=pd.RangeIndex(len(encoded), name="transaction_id"),
    )


# Alias kept for users who expect the "encode_one_hot" spelling.
encode_one_hot = one_hot_encode


def _as_itemset(itemset: Union[str, Iterable[str]]) -> Itemset:
    """Convert a string or iterable into a frozenset of string items."""
    if itemset is None:
        raise ValueError("An itemset must not be None")
    if isinstance(itemset, str):
        return frozenset([itemset])
    try:
        return frozenset(item if isinstance(item, str) else str(item)
                         for item in itemset)
    except TypeError as exc:
        raise ValueError("An itemset must be a string or an iterable of items") from exc


def _ensure_encoded(transactions: TransactionInput) -> List[FrozenSet[str]]:
    """Avoid re-encoding a list that already contains frozensets."""
    if isinstance(transactions, list):
        if not transactions:
            return []
        if all(isinstance(basket, frozenset) for basket in transactions):
            return transactions
    return encode_transactions(transactions)


def _sorted_items(itemset: Itemset) -> Tuple[str, ...]:
    """Return items in a stable order (handles non-string items defensively)."""
    return tuple(sorted(itemset, key=lambda item: str(item)))


def _itemset_sort_key(itemset: Itemset) -> Tuple[int, Tuple[str, ...]]:
    return (len(itemset), _sorted_items(itemset))


# ---------------------------------------------------------------------------
# Basic metrics
# ---------------------------------------------------------------------------
def support_count(transactions: TransactionInput, itemset: Union[str, Iterable[str]]) -> int:
    """Return the absolute number of transactions containing ``itemset``."""
    encoded = _ensure_encoded(transactions)
    items = _as_itemset(itemset)
    if not items:
        return len(encoded)
    return sum(1 for basket in encoded if items.issubset(basket))


def support(transactions: TransactionInput, itemset: Union[str, Iterable[str]]) -> float:
    """
    Relative support of an itemset: ``count(itemset) / n_transactions``.

    The empty itemset has support 1.0 by convention.
    """
    encoded = _ensure_encoded(transactions)
    items = _as_itemset(itemset)
    if not items:
        return 1.0
    if not encoded:
        return 0.0
    return support_count(encoded, items) / len(encoded)


def rule_metrics(
    transactions: TransactionInput,
    antecedent: Union[str, Iterable[str]],
    consequent: Union[str, Iterable[str]],
) -> Dict[str, float]:
    """
    Compute all standard metrics for a single rule ``antecedent -> consequent``.

    Returns a dictionary with ``antecedent_support``, ``consequent_support``,
    ``support`` (of the union), ``confidence``, ``lift``, ``leverage`` and
    ``conviction``.
    """
    encoded = _ensure_encoded(transactions)
    antecedent = _as_itemset(antecedent)
    consequent = _as_itemset(consequent)

    if not antecedent:
        raise ValueError("The antecedent must contain at least one item")
    if not consequent:
        raise ValueError("The consequent must contain at least one item")
    if antecedent & consequent:
        raise ValueError("Antecedent and consequent must be disjoint")

    n = len(encoded)
    if n == 0:
        raise ValueError("Cannot compute rule metrics from an empty transaction list")

    union = antecedent | consequent
    ant_count = support_count(encoded, antecedent)
    con_count = support_count(encoded, consequent)
    union_count = support_count(encoded, union)

    if ant_count == 0:
        raise ValueError("The antecedent has zero support; the rule is undefined")
    if con_count == 0:
        raise ValueError("The consequent has zero support; lift is undefined")

    ant_support = ant_count / n
    con_support = con_count / n
    union_support = union_count / n
    confidence = union_support / ant_support
    lift = confidence / con_support
    leverage = union_support - ant_support * con_support

    if confidence >= 1.0 - 1e-12:
        conviction = 1.0 if con_support >= 1.0 - 1e-12 else inf
    else:
        conviction = (1.0 - con_support) / (1.0 - confidence)

    return {
        "antecedent_support": float(ant_support),
        "consequent_support": float(con_support),
        "support": float(union_support),
        "confidence": float(confidence),
        "lift": float(lift),
        "leverage": float(leverage),
        "conviction": float(conviction),
    }


def confidence(
    transactions: TransactionInput,
    antecedent: Union[str, Iterable[str]],
    consequent: Union[str, Iterable[str]],
) -> float:
    """Confidence of ``antecedent -> consequent``: support(X ∪ Y) / support(X)."""
    return rule_metrics(transactions, antecedent, consequent)["confidence"]


def lift(
    transactions: TransactionInput,
    antecedent: Union[str, Iterable[str]],
    consequent: Union[str, Iterable[str]],
) -> float:
    """Lift: confidence / support(consequent). Values above 1 mean positive association."""
    return rule_metrics(transactions, antecedent, consequent)["lift"]


def leverage(
    transactions: TransactionInput,
    antecedent: Union[str, Iterable[str]],
    consequent: Union[str, Iterable[str]],
) -> float:
    """Leverage: support(X ∪ Y) - support(X) * support(Y)."""
    return rule_metrics(transactions, antecedent, consequent)["leverage"]


def conviction(
    transactions: TransactionInput,
    antecedent: Union[str, Iterable[str]],
    consequent: Union[str, Iterable[str]],
) -> float:
    """
    Conviction: ``(1 - support(Y)) / (1 - confidence)``.

    Perfect rules (confidence 1) have infinite conviction unless the consequent
    also has support 1, in which case the ratio is defined as 1.0.
    """
    return rule_metrics(transactions, antecedent, consequent)["conviction"]


def itemsets_to_dataframe(frequent_itemsets: Mapping[Itemset, float]) -> pd.DataFrame:
    """Convert a ``{itemset: support}`` mapping into a tidy DataFrame."""
    if isinstance(frequent_itemsets, pd.DataFrame):
        missing = {"support", "itemsets"} - set(frequent_itemsets.columns)
        if missing:
            raise ValueError(f"DataFrame is missing columns: {sorted(missing)}")
        frame = frequent_itemsets.loc[:, ["support", "itemsets"]].copy()
        frame["itemsets"] = frame["itemsets"].map(frozenset)
        frame = frame.sort_values(
            by="support", ascending=False, kind="mergesort"
        ).reset_index(drop=True)
        return frame

    if not isinstance(frequent_itemsets, Mapping):
        raise TypeError("frequent_itemsets must be a mapping or a DataFrame")

    rows = [
        (float(value), itemset if isinstance(itemset, frozenset) else frozenset(itemset))
        for itemset, value in frequent_itemsets.items()
    ]
    rows.sort(key=lambda row: _itemset_sort_key(row[1]))
    return pd.DataFrame(rows, columns=["support", "itemsets"])


# ---------------------------------------------------------------------------
# Apriori
# ---------------------------------------------------------------------------
class Apriori:
    """
    Apriori frequent-itemset miner.

    The algorithm works level by level. At level ``k`` it joins pairs of
    frequent ``(k-1)``-itemsets that share the same first ``k-2`` items, then
    removes candidates that contain an infrequent ``(k-1)``-subset. That
    anti-monotone prune is safe: if a subset is infrequent, every superset is
    infrequent too.

    Parameters
    ----------
    min_support : float, default=0.01
        Minimum relative support in ``(0, 1]``.
    max_len : int or None, default=None
        Optional maximum itemset size.

    Attributes
    ----------
    frequent_itemsets_ : dict of frozenset -> float
        Frequent itemsets and their relative supports.
    support_counts_ : dict of frozenset -> int
        Absolute support counts.
    n_transactions_ : int
        Number of transactions seen during ``fit``.
    items_ : list of str
        All distinct items in the dataset.
    level_stats_ : list of dict
        Per-level diagnostics with the number of joined candidates, pruned
        candidates, surviving candidates and frequent itemsets.
    """

    def __init__(self, min_support: float = 0.01, max_len: Optional[int] = None):
        self.min_support = float(min_support)
        self.max_len = max_len
        self._validate_parameters()

        self.frequent_itemsets_: Dict[Itemset, float] = {}
        self.support_counts_: Dict[Itemset, int] = {}
        self.n_transactions_: int = 0
        self.items_: List[str] = []
        self.level_stats_: List[Dict[str, int]] = []

    def _validate_parameters(self) -> None:
        if not np.isfinite(self.min_support) or not 0.0 < self.min_support <= 1.0:
            raise ValueError("min_support must be a number in (0, 1]")
        if self.max_len is not None:
            if not isinstance(self.max_len, (int, np.integer)) or self.max_len < 1:
                raise ValueError("max_len must be a positive integer or None")

    def fit(self, transactions: TransactionInput) -> "Apriori":
        """Mine all frequent itemsets from ``transactions``."""
        encoded = _ensure_encoded(transactions)
        n = len(encoded)
        self.n_transactions_ = n
        self.items_ = sorted({item for basket in encoded for item in basket})
        self.frequent_itemsets_ = {}
        self.support_counts_ = {}
        self.level_stats_ = []

        if n == 0:
            return self

        min_count = self.min_support * n
        tolerance = 1e-12

        item_counts = Counter(item for basket in encoded for item in basket)
        current_counts = {
            frozenset([item]): count
            for item, count in item_counts.items()
            if count + tolerance >= min_count
        }
        current_counts = dict(sorted(current_counts.items(), key=lambda kv: _itemset_sort_key(kv[0])))

        all_counts: Dict[Itemset, int] = dict(current_counts)
        k = 1

        while current_counts and (self.max_len is None or k < self.max_len):
            k += 1
            previous = sorted(current_counts.keys(), key=_itemset_sort_key)

            # --- Join step: combine itemsets with an identical (k-2)-prefix ---
            groups: Dict[Tuple[str, ...], List[Tuple[str, ...]]] = defaultdict(list)
            for itemset in previous:
                ordered = _sorted_items(itemset)
                groups[ordered[:-1]].append(ordered)

            joined: set = set()
            for prefix, group in groups.items():
                for left, right in combinations(group, 2):
                    joined.add(frozenset(prefix + (left[-1], right[-1])))

            # --- Anti-monotone prune: every (k-1)-subset must be frequent ---
            previous_set = set(previous)
            valid_candidates: List[Itemset] = []
            n_pruned = 0
            for candidate in sorted(joined, key=_itemset_sort_key):
                candidate_items = _sorted_items(candidate)
                subsets_are_frequent = True
                for subset in combinations(candidate_items, k - 1):
                    if frozenset(subset) not in previous_set:
                        subsets_are_frequent = False
                        break
                if subsets_are_frequent:
                    valid_candidates.append(candidate)
                else:
                    n_pruned += 1

            # --- Count candidates by scanning the transactions once ---
            valid_set = set(valid_candidates)
            candidate_counts: Counter = Counter()
            for basket in encoded:
                if len(basket) < k:
                    continue
                for combination in combinations(_sorted_items(basket), k):
                    candidate = frozenset(combination)
                    if candidate in valid_set:
                        candidate_counts[candidate] += 1

            next_counts = {
                candidate: count
                for candidate, count in candidate_counts.items()
                if count + tolerance >= min_count
            }
            next_counts = dict(sorted(next_counts.items(), key=lambda kv: _itemset_sort_key(kv[0])))

            self.level_stats_.append(
                {
                    "level": k,
                    "joined_candidates": len(joined),
                    "pruned_candidates": n_pruned,
                    "candidates": len(valid_candidates),
                    "frequent": len(next_counts),
                }
            )

            all_counts.update(next_counts)
            current_counts = next_counts

        all_counts = dict(sorted(all_counts.items(), key=lambda kv: _itemset_sort_key(kv[0])))
        self.support_counts_ = all_counts
        self.frequent_itemsets_ = {
            itemset: count / n for itemset, count in all_counts.items()
        }
        return self

    def get_support(self, itemset: Union[str, Iterable[str]]) -> float:
        """Return the support of ``itemset`` (0.0 if it was not frequent)."""
        return float(self.frequent_itemsets_.get(_as_itemset(itemset), 0.0))

    def to_dataframe(self) -> pd.DataFrame:
        """Return the frequent itemsets as a ``support``/``itemsets`` DataFrame."""
        return itemsets_to_dataframe(self.frequent_itemsets_)


def apriori(
    transactions: TransactionInput,
    min_support: float = 0.01,
    max_len: Optional[int] = None,
    as_dataframe: bool = False,
) -> Union[Dict[Itemset, float], pd.DataFrame]:
    """
    Convenience wrapper around :class:`Apriori`.

    Returns a ``{itemset: support}`` dictionary by default, or a tidy DataFrame
    when ``as_dataframe=True``.
    """
    model = Apriori(min_support=min_support, max_len=max_len).fit(transactions)
    if as_dataframe:
        return model.to_dataframe()
    return dict(model.frequent_itemsets_)


# ---------------------------------------------------------------------------
# FP-Growth
# ---------------------------------------------------------------------------
class _FPNode:
    """A node in an FP-tree."""

    __slots__ = ("item", "count", "parent", "children")

    def __init__(self, item: Optional[str], parent: Optional["_FPNode"]):
        self.item = item
        self.count: float = 0.0
        self.parent = parent
        self.children: Dict[str, "_FPNode"] = {}


def _build_fp_tree(
    weighted_transactions: Sequence[Tuple[Itemset, float]],
    min_count: float,
) -> Tuple[_FPNode, Dict[str, List[_FPNode]], Dict[str, float]]:
    """
    Build an FP-tree from ``(itemset, weight)`` pairs.

    Returns the root node, the header table (item -> list of nodes) and the
    support count of every item that survives ``min_count``.
    """
    counts: Counter = Counter()
    normalized: List[Tuple[Itemset, float]] = []
    for itemset, weight in weighted_transactions:
        if weight <= 0:
            continue
        items = frozenset(itemset)
        if not items:
            continue
        normalized.append((items, float(weight)))
        for item in items:
            counts[item] += float(weight)

    frequent = {
        item: count for item, count in counts.items()
        if count + 1e-12 >= min_count
    }
    root = _FPNode(None, None)
    if not frequent:
        return root, {}, {}

    order = {
        item: rank
        for rank, (item, _) in enumerate(
            sorted(frequent.items(), key=lambda kv: (-kv[1], str(kv[0])))
        )
    }
    header: Dict[str, List[_FPNode]] = {item: [] for item in order}

    for itemset, weight in normalized:
        ordered_items = [item for item in itemset if item in order]
        if not ordered_items:
            continue
        ordered_items.sort(key=lambda item: order[item])

        node = root
        for item in ordered_items:
            child = node.children.get(item)
            if child is None:
                child = _FPNode(item, node)
                node.children[item] = child
                header[item].append(child)
            child.count += weight
            node = child

    return root, header, dict(frequent)


class FPGrowth:
    """
    FP-Growth frequent-itemset miner.

    The first pass builds an FP-tree, a compressed prefix tree of the baskets.
    Mining then processes each item in the header table, builds its conditional
    pattern base, and recursively mines a conditional FP-tree. No candidate
    generation is needed.

    Parameters
    ----------
    min_support : float, default=0.01
        Minimum relative support in ``(0, 1]``.
    max_len : int or None, default=None
        Optional maximum itemset size.

    Attributes
    ----------
    frequent_itemsets_ : dict of frozenset -> float
        Frequent itemsets and their relative supports.
    support_counts_ : dict of frozenset -> int
        Absolute support counts.
    n_transactions_ : int
        Number of transactions seen during ``fit``.
    items_ : list of str
        All distinct items in the dataset.
    tree_ : _FPNode
        Root of the fitted FP-tree.
    header_ : dict of str -> list of _FPNode
        Header table with the node links for each frequent item.
    """

    def __init__(self, min_support: float = 0.01, max_len: Optional[int] = None):
        self.min_support = float(min_support)
        self.max_len = max_len
        self._validate_parameters()

        self.frequent_itemsets_: Dict[Itemset, float] = {}
        self.support_counts_: Dict[Itemset, int] = {}
        self.n_transactions_: int = 0
        self.items_: List[str] = []
        self.tree_: Optional[_FPNode] = None
        self.header_: Dict[str, List[_FPNode]] = {}

    def _validate_parameters(self) -> None:
        if not np.isfinite(self.min_support) or not 0.0 < self.min_support <= 1.0:
            raise ValueError("min_support must be a number in (0, 1]")
        if self.max_len is not None:
            if not isinstance(self.max_len, (int, np.integer)) or self.max_len < 1:
                raise ValueError("max_len must be a positive integer or None")

    def fit(self, transactions: TransactionInput) -> "FPGrowth":
        """Mine all frequent itemsets with FP-Growth."""
        encoded = _ensure_encoded(transactions)
        n = len(encoded)
        self.n_transactions_ = n
        self.items_ = sorted({item for basket in encoded for item in basket})
        self.frequent_itemsets_ = {}
        self.support_counts_ = {}
        self.tree_ = _FPNode(None, None)
        self.header_ = {}

        if n == 0:
            return self

        min_count = self.min_support * n
        weighted = [(basket, 1.0) for basket in encoded]
        root, header, counts = _build_fp_tree(weighted, min_count)
        self.tree_ = root
        self.header_ = header

        results: Dict[Itemset, int] = {}
        self._mine(frozenset(), header, counts, min_count, results)

        all_counts = dict(sorted(results.items(), key=lambda kv: _itemset_sort_key(kv[0])))
        self.support_counts_ = all_counts
        self.frequent_itemsets_ = {
            itemset: count / n for itemset, count in all_counts.items()
        }
        return self

    def _mine(
        self,
        prefix: Itemset,
        header: Dict[str, List[_FPNode]],
        counts: Dict[str, float],
        min_count: float,
        results: Dict[Itemset, int],
    ) -> None:
        """Recursively mine conditional FP-trees."""
        for item in sorted(header, key=lambda name: (counts[name], str(name))):
            itemset = prefix | {item}
            results[itemset] = int(round(counts[item]))

            if self.max_len is not None and len(itemset) >= self.max_len:
                continue

            # Conditional pattern base: every prefix path that reaches ``item``.
            pattern_base: Counter = Counter()
            for node in header[item]:
                path: List[str] = []
                parent = node.parent
                while parent is not None and parent.item is not None:
                    path.append(parent.item)
                    parent = parent.parent
                if path:
                    pattern_base[frozenset(path)] += node.count

            if not pattern_base:
                continue

            weighted_paths = [(itemset_, weight) for itemset_, weight in pattern_base.items()]
            _, conditional_header, conditional_counts = _build_fp_tree(
                weighted_paths, min_count
            )
            if conditional_header:
                self._mine(
                    itemset,
                    conditional_header,
                    conditional_counts,
                    min_count,
                    results,
                )

    def get_support(self, itemset: Union[str, Iterable[str]]) -> float:
        """Return the support of ``itemset`` (0.0 if it was not frequent)."""
        return float(self.frequent_itemsets_.get(_as_itemset(itemset), 0.0))

    def to_dataframe(self) -> pd.DataFrame:
        """Return the frequent itemsets as a ``support``/``itemsets`` DataFrame."""
        return itemsets_to_dataframe(self.frequent_itemsets_)


def fpgrowth(
    transactions: TransactionInput,
    min_support: float = 0.01,
    max_len: Optional[int] = None,
    as_dataframe: bool = False,
) -> Union[Dict[Itemset, float], pd.DataFrame]:
    """
    Convenience wrapper around :class:`FPGrowth`.

    Returns a ``{itemset: support}`` dictionary by default, or a tidy DataFrame
    when ``as_dataframe=True``.
    """
    model = FPGrowth(min_support=min_support, max_len=max_len).fit(transactions)
    if as_dataframe:
        return model.to_dataframe()
    return dict(model.frequent_itemsets_)


# ---------------------------------------------------------------------------
# Association-rule generation
# ---------------------------------------------------------------------------
def _coerce_frequent_itemsets(
    frequent_itemsets: Union[Mapping[Itemset, float], pd.DataFrame]
) -> Dict[Itemset, float]:
    """Normalize a mapping or DataFrame of frequent itemsets."""
    if isinstance(frequent_itemsets, pd.DataFrame):
        missing = {"support", "itemsets"} - set(frequent_itemsets.columns)
        if missing:
            raise ValueError(f"DataFrame is missing columns: {sorted(missing)}")
        return {
            frozenset(itemset): float(value)
            for itemset, value in zip(
                frequent_itemsets["itemsets"], frequent_itemsets["support"]
            )
        }

    if not isinstance(frequent_itemsets, Mapping):
        raise TypeError("frequent_itemsets must be a mapping or a DataFrame")
    return {
        frozenset(itemset): float(value)
        for itemset, value in frequent_itemsets.items()
    }


def _conviction(confidence_value: float, consequent_support: float) -> float:
    if confidence_value >= 1.0 - 1e-12:
        return 1.0 if consequent_support >= 1.0 - 1e-12 else inf
    return (1.0 - consequent_support) / (1.0 - confidence_value)


def generate_rules(
    frequent_itemsets: Union[Mapping[Itemset, float], pd.DataFrame],
    min_confidence: float = 0.5,
    min_lift: float = 0.0,
) -> pd.DataFrame:
    """
    Generate association rules from frequent itemsets.

    Every frequent itemset of size ``k >= 2`` is split into a non-empty
    antecedent and a non-empty consequent. A rule is kept when its confidence is
    at least ``min_confidence`` (and its lift is at least ``min_lift``).

    Parameters
    ----------
    frequent_itemsets : mapping or DataFrame
        Either ``{itemset: support}`` or a DataFrame with ``support`` and
        ``itemsets`` columns.
    min_confidence : float, default=0.5
        Minimum confidence threshold in ``[0, 1]``.
    min_lift : float, default=0.0
        Minimum lift threshold.

    Returns
    -------
    pandas.DataFrame
        Columns: ``antecedents``, ``consequents``, ``antecedent_support``,
        ``consequent_support``, ``support``, ``confidence``, ``lift``,
        ``leverage``, ``conviction``. Sorted by confidence and lift.
    """
    min_confidence = float(min_confidence)
    min_lift = float(min_lift)
    if not np.isfinite(min_confidence) or not 0.0 <= min_confidence <= 1.0:
        raise ValueError("min_confidence must be a number in [0, 1]")
    if not np.isfinite(min_lift):
        raise ValueError("min_lift must be a finite number")

    supports = _coerce_frequent_itemsets(frequent_itemsets)
    rows: List[Dict[str, object]] = []

    for itemset, union_support in supports.items():
        itemset = frozenset(itemset)
        if len(itemset) < 2:
            continue

        ordered = _sorted_items(itemset)
        for size in range(1, len(ordered)):
            for antecedent_items in combinations(ordered, size):
                antecedent = frozenset(antecedent_items)
                consequent = itemset - antecedent
                antecedent_support = supports.get(antecedent)
                consequent_support = supports.get(consequent)

                # Anti-monotonicity guarantees these are present, but a caller
                # may pass a partial mapping.
                if antecedent_support is None or consequent_support is None:
                    continue
                if antecedent_support <= 0.0 or consequent_support <= 0.0:
                    continue

                confidence_value = union_support / antecedent_support
                if confidence_value + 1e-12 < min_confidence:
                    continue

                lift_value = confidence_value / consequent_support
                if lift_value + 1e-12 < min_lift:
                    continue

                leverage_value = (
                    union_support - antecedent_support * consequent_support
                )
                rows.append(
                    {
                        "antecedents": antecedent,
                        "consequents": consequent,
                        "antecedent_support": float(antecedent_support),
                        "consequent_support": float(consequent_support),
                        "support": float(union_support),
                        "confidence": float(confidence_value),
                        "lift": float(lift_value),
                        "leverage": float(leverage_value),
                        "conviction": float(
                            _conviction(confidence_value, consequent_support)
                        ),
                    }
                )

    frame = pd.DataFrame(rows, columns=RULES_COLUMNS)
    if not frame.empty:
        frame = frame.sort_values(
            by=["confidence", "lift", "support"],
            ascending=False,
            kind="mergesort",
        ).reset_index(drop=True)
    return frame


# Common spelling used by mlxtend and by many tutorials.
association_rules = generate_rules


# ---------------------------------------------------------------------------
# Synthetic warung / minimarket baskets
# ---------------------------------------------------------------------------
WARUNG_PRODUCTS = (
    "Indomie Goreng",
    "Mi Instan Kuah",
    "Telur Ayam",
    "Beras",
    "Minyak Goreng",
    "Gula Pasir",
    "Kopi Sachet",
    "Teh Celup",
    "Rokok",
    "Korek Api",
    "Roti Tawar",
    "Susu UHT",
    "Air Mineral",
    "Sabun Mandi",
    "Shampo Sachet",
    "Biskuit",
    "Keripik",
    "Es Krim",
)

# Pairs that are deliberately over-represented in the synthetic baskets.
PLANTED_ASSOCIATIONS = (
    ("Indomie Goreng", "Telur Ayam"),
    ("Kopi Sachet", "Gula Pasir"),
    ("Rokok", "Korek Api"),
    ("Roti Tawar", "Susu UHT"),
)

_BASE_WEIGHTS = {
    "Indomie Goreng": 0.10,
    "Mi Instan Kuah": 0.06,
    "Telur Ayam": 0.09,
    "Beras": 0.06,
    "Minyak Goreng": 0.05,
    "Gula Pasir": 0.06,
    "Kopi Sachet": 0.08,
    "Teh Celup": 0.05,
    "Rokok": 0.10,
    "Korek Api": 0.04,
    "Roti Tawar": 0.07,
    "Susu UHT": 0.07,
    "Air Mineral": 0.12,
    "Sabun Mandi": 0.04,
    "Shampo Sachet": 0.03,
    "Biskuit": 0.05,
    "Keripik": 0.05,
    "Es Krim": 0.03,
}


def generate_warung_transactions(
    n_transactions: int = 1000,
    random_state: Optional[int] = 42,
) -> List[List[str]]:
    """
    Generate synthetic Indonesian warung/minimarket baskets.

    Each basket starts from a few random products and is then augmented with
    planted associations: Indomie + telur, kopi sachet + gula, rokok + korek,
    and roti + susu. The planted pairs make association-rule mining produce
    meaningful, easy-to-interpret results.

    Parameters
    ----------
    n_transactions : int, default=1000
        Number of baskets to generate.
    random_state : int or None, default=42
        Seed for reproducibility.

    Returns
    -------
    list of list of str
        One list of product names per transaction, in catalog order.
    """
    if not isinstance(n_transactions, (int, np.integer)):
        raise ValueError("n_transactions must be an integer")
    if n_transactions < 0:
        raise ValueError("n_transactions must be non-negative")

    rng = np.random.default_rng(random_state)
    products = list(WARUNG_PRODUCTS)
    weights = np.array([_BASE_WEIGHTS[product] for product in products], dtype=float)
    weights /= weights.sum()

    baskets: List[List[str]] = []
    for _ in range(int(n_transactions)):
        basket: set = set()

        # Plant at most one target association per basket. Keeping the pairs
        # separate avoids accidental cross-pair rules in the demo.
        if rng.random() < 0.70:
            pair_index = int(rng.integers(0, len(PLANTED_ASSOCIATIONS)))
            basket.update(PLANTED_ASSOCIATIONS[pair_index])

        # Fill the basket with a few random products.
        n_random = int(rng.integers(1, 4))
        picks = rng.choice(products, size=n_random, replace=False, p=weights)
        basket.update(str(item) for item in picks)

        if not basket:  # defensive: n_random is always at least one
            basket.add("Air Mineral")

        baskets.append([product for product in products if product in basket])

    return baskets


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------
def _format_itemset(itemset: Itemset) -> str:
    return "{" + ", ".join(_sorted_items(itemset)) + "}"


def _demo() -> None:
    print("Association Rule Mining - Warung Demo")
    print("=" * 72)

    transactions = generate_warung_transactions(n_transactions=800, random_state=42)
    one_hot = one_hot_encode(transactions)

    print(f"Transactions : {len(transactions)}")
    print(f"Distinct items: {len(one_hot.columns)}")
    print(f"Basket shape  : {one_hot.shape} (one-hot matrix)")
    avg_basket = float(one_hot.sum(axis=1).mean()) if len(one_hot.columns) else 0.0
    print(f"Average basket: {avg_basket:.2f} items")
    print("\nSample baskets:")
    for basket in transactions[:5]:
        print("  -", ", ".join(basket))

    min_support = 0.02
    apriori_model = Apriori(min_support=min_support).fit(transactions)
    fp_model = FPGrowth(min_support=min_support).fit(transactions)

    print(f"\nFrequent itemsets (min_support={min_support}):")
    print(f"  Apriori  : {len(apriori_model.frequent_itemsets_)}")
    print(f"  FP-Growth: {len(fp_model.frequent_itemsets_)}")
    print(f"  Identical: {apriori_model.frequent_itemsets_ == fp_model.frequent_itemsets_}")

    print("\nTop 10 frequent itemsets (Apriori):")
    table = apriori_model.to_dataframe().sort_values(
        by=["support", "itemsets"], ascending=[False, True]
    ).head(10)
    for _, row in table.iterrows():
        print(f"  {row['support']:.3f}  {_format_itemset(row['itemsets'])}")

    print("\nTop 10 rules by lift (min_confidence=0.50):")
    rules = generate_rules(apriori_model.frequent_itemsets_, min_confidence=0.50)
    rules = rules.sort_values(
        by=["lift", "confidence"], ascending=False
    ).head(10)
    for _, row in rules.iterrows():
        print(
            f"  {_format_itemset(row['antecedents'])} -> "
            f"{_format_itemset(row['consequents'])}  "
            f"sup={row['support']:.3f}  conf={row['confidence']:.3f}  "
            f"lift={row['lift']:.2f}"
        )

    print("\nPlanted associations (rule metrics):")
    for left, right in PLANTED_ASSOCIATIONS:
        metrics = rule_metrics(transactions, left, right)
        print(
            f"  {left:16s} -> {right:12s}  "
            f"conf={metrics['confidence']:.3f}  "
            f"lift={metrics['lift']:.2f}  "
            f"leverage={metrics['leverage']:+.3f}  "
            f"conviction={metrics['conviction']:.2f}"
        )

    print("\nDone. Try different min_support / min_confidence values!")


if __name__ == "__main__":
    _demo()
