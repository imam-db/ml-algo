"""
Association Rules with mlxtend
==============================

This script compares the from-scratch implementations in ``implementation.py``
with the reference implementations in ``mlxtend``:

- ``mlxtend.frequent_patterns.apriori`` vs our :class:`Apriori`,
- ``mlxtend.frequent_patterns.fpgrowth`` vs our :class:`FPGrowth`,
- ``mlxtend.frequent_patterns.association_rules`` vs our
  :func:`generate_rules`.

``mlxtend`` is an optional dependency. If it is not installed, the script
prints a friendly message and exits successfully instead of failing.

Run from the repository root:

    python3 02_unsupervised_learning/06_association_rules/sklearn_example.py
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, Mapping, Tuple

import numpy as np
import pandas as pd

# Allow both "python3 path/to/sklearn_example.py" and execution from elsewhere.
try:
    from implementation import (
        Apriori,
        FPGrowth,
        PLANTED_ASSOCIATIONS,
        apriori,
        fpgrowth,
        generate_rules,
        generate_warung_transactions,
        one_hot_encode,
        rule_metrics,
    )
except ImportError:  # pragma: no cover - only relevant to unusual launch modes
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from implementation import (  # type: ignore
        Apriori,
        FPGrowth,
        PLANTED_ASSOCIATIONS,
        apriori,
        fpgrowth,
        generate_rules,
        generate_warung_transactions,
        one_hot_encode,
        rule_metrics,
    )


try:
    from mlxtend.frequent_patterns import (
        apriori as ml_apriori,
        association_rules as ml_association_rules,
        fpgrowth as ml_fpgrowth,
    )
    from mlxtend.preprocessing import TransactionEncoder

    MLXTEND_AVAILABLE = True
except ImportError:
    MLXTEND_AVAILABLE = False


Itemset = frozenset
RuleKey = Tuple[Itemset, Itemset]


def _frequent_map(frame: pd.DataFrame) -> Dict[Itemset, float]:
    """Convert an mlxtend frequent-itemset DataFrame to ``{itemset: support}``."""
    return {
        frozenset(itemset): float(value)
        for itemset, value in zip(frame["itemsets"], frame["support"])
    }


def _rules_map(frame: pd.DataFrame) -> Dict[RuleKey, Dict[str, float]]:
    """Convert a rule DataFrame to a dictionary keyed by (antecedent, consequent)."""
    # mlxtend uses spaces ("antecedent support"); we use underscores.
    antecedent_column = (
        "antecedent_support"
        if "antecedent_support" in frame.columns
        else "antecedent support"
    )
    consequent_column = (
        "consequent_support"
        if "consequent_support" in frame.columns
        else "consequent support"
    )
    result: Dict[RuleKey, Dict[str, float]] = {}
    for _, row in frame.iterrows():
        key = (frozenset(row["antecedents"]), frozenset(row["consequents"]))
        result[key] = {
            "antecedent_support": float(row[antecedent_column]),
            "consequent_support": float(row[consequent_column]),
            "support": float(row["support"]),
            "confidence": float(row["confidence"]),
            "lift": float(row["lift"]),
            "leverage": float(row["leverage"]),
            "conviction": float(row["conviction"]),
        }
    return result


def _compare_frequent(
    ours: Mapping[Itemset, float],
    theirs: Mapping[Itemset, float],
    label: str,
    reference_label: str = "mlxtend",
) -> bool:
    ours_keys = set(ours)
    theirs_keys = set(theirs)
    same_keys = ours_keys == theirs_keys
    max_difference = 0.0
    if same_keys:
        max_difference = max(
            (abs(ours[key] - theirs[key]) for key in ours_keys), default=0.0
        )

    print(f"\n{label}")
    print(f"  our itemsets : {len(ours)}")
    print(f"  mlxtend      : {len(theirs)}")
    print(f"  same itemsets: {same_keys}")
    print(f"  max |Δsupport|: {max_difference:.2e}")

    if not same_keys:
        print(f"  only ours    : {list(ours_keys - theirs_keys)[:5]}")
        print(f"  only mlxtend : {list(theirs_keys - ours_keys)[:5]}")
    return same_keys and max_difference <= 1e-9


def _compare_rules(
    ours: Mapping[RuleKey, Dict[str, float]],
    theirs: Mapping[RuleKey, Dict[str, float]],
) -> bool:
    same_keys = set(ours) == set(theirs)
    metrics = (
        "support",
        "confidence",
        "lift",
        "leverage",
        "antecedent_support",
        "consequent_support",
    )
    max_differences = {metric: 0.0 for metric in metrics}
    conviction_mismatches = 0

    if same_keys:
        for key in ours:
            for metric in metrics:
                max_differences[metric] = max(
                    max_differences[metric],
                    abs(ours[key][metric] - theirs[key][metric]),
                )
            ours_conviction = ours[key]["conviction"]
            theirs_conviction = theirs[key]["conviction"]
            if np.isinf(ours_conviction) or np.isinf(theirs_conviction):
                if not (np.isinf(ours_conviction) and np.isinf(theirs_conviction)):
                    conviction_mismatches += 1
            elif not np.isclose(
                ours_conviction, theirs_conviction, rtol=1e-9, atol=1e-9
            ):
                conviction_mismatches += 1

    print("\nAssociation rules")
    print(f"  our rules            : {len(ours)}")
    print(f"  mlxtend rules        : {len(theirs)}")
    print(f"  same (antecedent, consequent): {same_keys}")
    for metric, difference in max_differences.items():
        print(f"  max |Δ{metric:18s}|: {difference:.2e}")
    print(f"  conviction mismatches: {conviction_mismatches}")

    if not same_keys:
        print(f"  only ours    : {list(set(ours) - set(theirs))[:5]}")
        print(f"  only mlxtend : {list(set(theirs) - set(ours))[:5]}")

    return (
        same_keys
        and conviction_mismatches == 0
        and all(difference <= 1e-9 for difference in max_differences.values())
    )


def _ml_association_rules(
    frequent: pd.DataFrame, min_confidence: float
) -> pd.DataFrame:
    """Call mlxtend across versions that may or may not need ``num_itemsets``."""
    try:
        return ml_association_rules(
            frequent,
            metric="confidence",
            min_threshold=min_confidence,
            num_itemsets=len(frequent),
        )
    except TypeError:  # older mlxtend releases
        return ml_association_rules(
            frequent, metric="confidence", min_threshold=min_confidence
        )


def main() -> int:
    if not MLXTEND_AVAILABLE:
        print("mlxtend is not installed; skipping the comparison.")
        print("Install it with:  pip install mlxtend")
        print("The from-scratch implementation still works without mlxtend.")
        return 0

    print("Association Rules - mlxtend comparison")
    print("=" * 72)

    n_transactions = 600
    min_support = 0.03
    min_confidence = 0.50

    transactions = generate_warung_transactions(
        n_transactions=n_transactions, random_state=42
    )
    print(f"Synthetic warung baskets: {n_transactions}")
    print(f"Average basket size      : {np.mean([len(b) for b in transactions]):.2f}")
    for left, right in PLANTED_ASSOCIATIONS:
        metrics = rule_metrics(transactions, left, right)
        print(
            f"  planted {left} + {right}: "
            f"support={metrics['support']:.3f}, lift={metrics['lift']:.2f}"
        )

    # --- Encode for mlxtend -------------------------------------------------
    encoder = TransactionEncoder()
    encoded_array = encoder.fit(transactions).transform(transactions)
    ml_matrix = pd.DataFrame(encoded_array, columns=encoder.columns_)
    our_matrix = one_hot_encode(transactions)
    matrices_agree = (
        list(our_matrix.columns) == list(ml_matrix.columns)
        and np.array_equal(our_matrix.to_numpy(), ml_matrix.to_numpy())
    )
    print(f"\nOne-hot matrices agree: {matrices_agree}")

    # --- Frequent itemsets --------------------------------------------------
    our_apriori = apriori(transactions, min_support=min_support)
    our_fpgrowth = fpgrowth(transactions, min_support=min_support)

    ml_apriori_frame = ml_apriori(
        ml_matrix, min_support=min_support, use_colnames=True
    )
    ml_fpgrowth_frame = ml_fpgrowth(
        ml_matrix, min_support=min_support, use_colnames=True
    )
    ml_apriori_map = _frequent_map(ml_apriori_frame)
    ml_fpgrowth_map = _frequent_map(ml_fpgrowth_frame)

    checks = [
        matrices_agree,
        _compare_frequent(our_apriori, ml_apriori_map, "Apriori vs mlxtend"),
        _compare_frequent(our_fpgrowth, ml_fpgrowth_map, "FP-Growth vs mlxtend"),
        _compare_frequent(
            our_apriori,
            our_fpgrowth,
            "Our Apriori vs our FP-Growth",
            reference_label="our FP-Growth",
        ),
    ]

    # Also check the class-based API matches the function-based API.
    class_model = Apriori(min_support=min_support).fit(transactions)
    fp_class_model = FPGrowth(min_support=min_support).fit(transactions)
    checks.append(
        class_model.frequent_itemsets_ == our_apriori
        and fp_class_model.frequent_itemsets_ == our_fpgrowth
    )

    # --- Association rules --------------------------------------------------
    our_rules_frame = generate_rules(
        our_apriori, min_confidence=min_confidence
    )
    ml_rules_frame = _ml_association_rules(
        ml_apriori_frame, min_confidence=min_confidence
    )
    checks.append(
        _compare_rules(
            _rules_map(our_rules_frame),
            _rules_map(ml_rules_frame),
        )
    )

    print("\n" + "=" * 72)
    if all(checks):
        print("All comparisons passed: our implementation matches mlxtend.")
        return 0

    print("Some comparisons FAILED. See the differences printed above.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
