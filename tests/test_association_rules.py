"""
Tests for the from-scratch Association Rules implementation.

The tests use a small hand-computed toy dataset so that every support,
confidence, lift, leverage and conviction value can be checked exactly. They
also compare Apriori with brute-force enumeration, compare FP-Growth with
Apriori, and exercise the synthetic warung data generator and edge cases.
"""

from itertools import combinations

import numpy as np
import pandas as pd
import pytest

import importlib.util
import os


# Load implementation.py exactly like tests/test_svm.py does.
spec = importlib.util.spec_from_file_location(
    "association_rules_impl",
    os.path.join(
        os.path.dirname(__file__),
        "..",
        "02_unsupervised_learning",
        "06_association_rules",
        "implementation.py",
    ),
)
ar = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ar)


# ---------------------------------------------------------------------------
# Shared toy dataset
# ---------------------------------------------------------------------------
TOY = [
    {"Bread", "Milk"},
    {"Bread", "Diaper", "Beer", "Eggs"},
    {"Milk", "Diaper", "Beer", "Cola"},
    {"Bread", "Milk", "Diaper", "Beer"},
    {"Bread", "Milk", "Diaper", "Cola"},
]
TOY_ITEMS = sorted({item for basket in TOY for item in basket})


def brute_force_itemsets(transactions, min_support):
    """Enumerate every non-empty itemset and keep those above the threshold."""
    encoded = [set(basket) for basket in transactions]
    n = len(encoded)
    if n == 0:
        return {}
    items = sorted({item for basket in encoded for item in basket})
    result = {}
    for size in range(1, len(items) + 1):
        for combo in combinations(items, size):
            itemset = frozenset(combo)
            count = sum(1 for basket in encoded if itemset <= basket)
            if count / n >= min_support - 1e-12:
                result[itemset] = count / n
    return result


# ---------------------------------------------------------------------------
# Encoding
# ---------------------------------------------------------------------------
def test_encode_transactions_normalizes_and_deduplicates():
    encoded = ar.encode_transactions(
        [["Bread", "Milk", "Milk"], [], ("Milk",)]
    )

    assert encoded == [
        frozenset({"Bread", "Milk"}),
        frozenset(),
        frozenset({"Milk"}),
    ]
    # A single string is one one-item basket, not an iterable of characters.
    assert ar.encode_transactions("Bread") == [frozenset({"Bread"})]


def test_one_hot_encode_matches_baskets():
    matrix = ar.one_hot_encode(TOY)

    assert matrix.shape == (5, len(TOY_ITEMS))
    assert list(matrix.columns) == TOY_ITEMS
    assert matrix.dtypes.eq(bool).all()

    # Row-wise comparison against the original baskets.
    for row_index, basket in enumerate(TOY):
        present = set(matrix.columns[matrix.iloc[row_index].to_numpy()])
        assert present == basket

    # DataFrame input is interpreted as an already-encoded matrix.
    frame = pd.DataFrame(
        [[True, False], [False, True], [True, True]],
        columns=["Bread", "Milk"],
    )
    assert ar.encode_transactions(frame) == [
        frozenset({"Bread"}),
        frozenset({"Milk"}),
        frozenset({"Bread", "Milk"}),
    ]


def test_encode_transactions_rejects_invalid_input():
    with pytest.raises(ValueError):
        ar.encode_transactions([1, 2, 3])  # scalars are not iterable baskets
    with pytest.raises(ValueError):
        ar.encode_transactions([["Bread", None]])


# ---------------------------------------------------------------------------
# Metrics on the hand-computed toy example
# ---------------------------------------------------------------------------
def test_support_values_are_exact():
    assert ar.support(TOY, {"Bread"}) == pytest.approx(0.8)
    assert ar.support(TOY, {"Milk"}) == pytest.approx(0.8)
    assert ar.support(TOY, {"Diaper"}) == pytest.approx(0.8)
    assert ar.support(TOY, {"Beer"}) == pytest.approx(0.6)
    assert ar.support(TOY, {"Eggs"}) == pytest.approx(0.2)
    assert ar.support(TOY, {"Cola"}) == pytest.approx(0.4)

    assert ar.support(TOY, {"Bread", "Milk"}) == pytest.approx(0.6)
    assert ar.support(TOY, {"Diaper", "Beer"}) == pytest.approx(0.6)
    assert ar.support(TOY, {"Bread", "Eggs"}) == pytest.approx(0.2)
    assert ar.support(TOY, {"Eggs", "Milk"}) == pytest.approx(0.0)
    assert ar.support_count(TOY, {"Diaper", "Beer"}) == 3

    # Empty itemset convention.
    assert ar.support(TOY, frozenset()) == pytest.approx(1.0)
    assert ar.support([], {"Bread"}) == pytest.approx(0.0)


def test_rule_metrics_hand_computed():
    bread_to_milk = ar.rule_metrics(TOY, "Bread", "Milk")
    assert bread_to_milk["antecedent_support"] == pytest.approx(0.8)
    assert bread_to_milk["consequent_support"] == pytest.approx(0.8)
    assert bread_to_milk["support"] == pytest.approx(0.6)
    assert bread_to_milk["confidence"] == pytest.approx(0.75)
    assert bread_to_milk["lift"] == pytest.approx(0.9375)
    assert bread_to_milk["leverage"] == pytest.approx(-0.04)
    assert bread_to_milk["conviction"] == pytest.approx(0.8)

    diaper_to_beer = ar.rule_metrics(TOY, {"Diaper"}, {"Beer"})
    assert diaper_to_beer["support"] == pytest.approx(0.6)
    assert diaper_to_beer["confidence"] == pytest.approx(0.75)
    assert diaper_to_beer["lift"] == pytest.approx(1.25)
    assert diaper_to_beer["leverage"] == pytest.approx(0.12)
    assert diaper_to_beer["conviction"] == pytest.approx(1.6)

    # Beer appears only when Diaper is present -> perfect confidence.
    beer_to_diaper = ar.rule_metrics(TOY, "Beer", "Diaper")
    assert beer_to_diaper["confidence"] == pytest.approx(1.0)
    assert beer_to_diaper["lift"] == pytest.approx(1.25)
    assert beer_to_diaper["leverage"] == pytest.approx(0.12)
    assert np.isinf(beer_to_diaper["conviction"])

    # The individual public functions must agree with rule_metrics.
    assert ar.confidence(TOY, "Diaper", "Beer") == pytest.approx(0.75)
    assert ar.lift(TOY, "Diaper", "Beer") == pytest.approx(1.25)
    assert ar.leverage(TOY, "Diaper", "Beer") == pytest.approx(0.12)
    assert ar.conviction(TOY, "Diaper", "Beer") == pytest.approx(1.6)


def test_rule_metrics_validation_and_edge_cases():
    # Disjoint items with no co-occurrence have zero confidence, not an error.
    assert ar.confidence(TOY, "Eggs", "Milk") == pytest.approx(0.0)

    # Use a genuinely absent item for the zero-support checks.
    with pytest.raises(ValueError, match="zero support"):
        ar.confidence(TOY, "Chocolate", "Milk")
    with pytest.raises(ValueError, match="zero support"):
        ar.lift(TOY, "Bread", "Chocolate")

    with pytest.raises(ValueError, match="disjoint"):
        ar.confidence(TOY, {"Bread"}, {"Bread", "Milk"})
    with pytest.raises(ValueError, match="antecedent"):
        ar.confidence(TOY, frozenset(), {"Milk"})
    with pytest.raises(ValueError, match="consequent"):
        ar.confidence(TOY, {"Milk"}, frozenset())
    with pytest.raises(ValueError, match="empty"):
        ar.confidence([], "Bread", "Milk")


# ---------------------------------------------------------------------------
# Apriori correctness and pruning
# ---------------------------------------------------------------------------
def test_apriori_matches_brute_force_enumeration():
    min_support = 0.2
    model = ar.Apriori(min_support=min_support).fit(TOY)
    expected = brute_force_itemsets(TOY, min_support)

    assert set(model.frequent_itemsets_) == set(expected)
    for itemset, expected_support in expected.items():
        assert model.frequent_itemsets_[itemset] == pytest.approx(expected_support)
        assert model.support_counts_[itemset] == round(expected_support * len(TOY))

    # The convenience function returns the same mapping.
    function_result = ar.apriori(TOY, min_support=min_support)
    assert function_result == model.frequent_itemsets_

    # All six items pass the 0.2 threshold on the toy data.
    assert {frozenset([item]) for item in TOY_ITEMS} <= set(model.frequent_itemsets_)


def test_apriori_support_boundary_is_inclusive():
    # Eggs has support exactly 0.2.
    inclusive = ar.Apriori(min_support=0.2).fit(TOY)
    assert frozenset({"Eggs"}) in inclusive.frequent_itemsets_

    exclusive = ar.Apriori(min_support=0.2000001).fit(TOY)
    assert frozenset({"Eggs"}) not in exclusive.frequent_itemsets_


def test_apriori_anti_monotone_property_and_subset_pruning():
    model = ar.Apriori(min_support=0.2).fit(TOY)
    frequent = set(model.frequent_itemsets_)

    # Anti-monotonicity: every subset of a frequent itemset is frequent.
    for itemset in frequent:
        for size in range(1, len(itemset)):
            for subset in combinations(itemset, size):
                assert frozenset(subset) in frequent

    # Equivalently, an infrequent itemset has no frequent superset.
    all_itemsets = set(brute_force_itemsets(TOY, 0.0))
    infrequent = all_itemsets - frequent
    for itemset in infrequent:
        for superset in all_itemsets:
            if itemset < superset:
                assert superset not in frequent

    # The subset check must actually remove candidates at some level.
    assert any(stat["pruned_candidates"] > 0 for stat in model.level_stats_)
    assert all(
        stat["candidates"] <= stat["joined_candidates"]
        for stat in model.level_stats_
    )


def test_apriori_respects_max_len():
    model = ar.Apriori(min_support=0.2, max_len=2).fit(TOY)
    assert model.frequent_itemsets_
    assert max(len(itemset) for itemset in model.frequent_itemsets_) == 2

    all_frequent = ar.Apriori(min_support=0.2).fit(TOY).frequent_itemsets_
    assert model.frequent_itemsets_ == {
        itemset: value
        for itemset, value in all_frequent.items()
        if len(itemset) <= 2
    }


@pytest.mark.slow
def test_apriori_matches_brute_force_on_generated_data():
    transactions = ar.generate_warung_transactions(180, random_state=7)
    min_support = 0.05
    model = ar.Apriori(min_support=min_support).fit(transactions)
    expected = brute_force_itemsets(transactions, min_support)

    assert set(model.frequent_itemsets_) == set(expected)
    for itemset, value in expected.items():
        assert model.frequent_itemsets_[itemset] == pytest.approx(value)


# ---------------------------------------------------------------------------
# FP-Growth correctness
# ---------------------------------------------------------------------------
def test_fpgrowth_matches_apriori_on_toy_data():
    apriori_model = ar.Apriori(min_support=0.2).fit(TOY)
    fp_model = ar.FPGrowth(min_support=0.2).fit(TOY)

    assert set(fp_model.frequent_itemsets_) == set(apriori_model.frequent_itemsets_)
    for itemset, value in apriori_model.frequent_itemsets_.items():
        assert fp_model.frequent_itemsets_[itemset] == pytest.approx(value)
        assert fp_model.support_counts_[itemset] == apriori_model.support_counts_[itemset]

    # Structure of the fitted FP-tree.
    assert fp_model.tree_ is not None
    assert fp_model.tree_.item is None
    assert fp_model.header_
    assert fp_model.n_transactions_ == len(TOY)


def test_fpgrowth_matches_apriori_on_generated_data():
    transactions = ar.generate_warung_transactions(350, random_state=21)

    for min_support in (0.02, 0.05, 0.10):
        apriori_model = ar.Apriori(min_support=min_support).fit(transactions)
        fp_model = ar.FPGrowth(min_support=min_support).fit(transactions)

        assert set(fp_model.frequent_itemsets_) == set(
            apriori_model.frequent_itemsets_
        )
        for itemset, value in apriori_model.frequent_itemsets_.items():
            assert fp_model.frequent_itemsets_[itemset] == pytest.approx(value)


def test_fpgrowth_matches_apriori_with_max_len():
    transactions = ar.generate_warung_transactions(150, random_state=3)

    apriori_result = ar.apriori(
        transactions, min_support=0.05, max_len=2
    )
    fp_result = ar.fpgrowth(transactions, min_support=0.05, max_len=2)

    assert apriori_result == fp_result
    assert max(len(itemset) for itemset in fp_result) <= 2


# ---------------------------------------------------------------------------
# Rule generation
# ---------------------------------------------------------------------------
def test_generate_rules_respects_min_confidence_and_metrics():
    frequent = ar.apriori(TOY, min_support=0.2)
    rules = ar.generate_rules(frequent, min_confidence=0.75)

    assert list(rules.columns) == ar.RULES_COLUMNS
    assert not rules.empty
    assert (rules["confidence"] >= 0.75 - 1e-12).all()

    # Every stored metric must agree with the direct metric functions.
    for _, row in rules.iterrows():
        expected = ar.rule_metrics(TOY, row["antecedents"], row["consequents"])
        assert row["support"] == pytest.approx(expected["support"])
        assert row["confidence"] == pytest.approx(expected["confidence"])
        assert row["lift"] == pytest.approx(expected["lift"])
        assert row["leverage"] == pytest.approx(expected["leverage"])
        if np.isinf(expected["conviction"]):
            assert np.isinf(row["conviction"])
        else:
            assert row["conviction"] == pytest.approx(expected["conviction"])

    # A specific hand-computed rule.
    diaper_beer = rules[
        (rules["antecedents"] == frozenset({"Diaper"}))
        & (rules["consequents"] == frozenset({"Beer"}))
    ]
    assert len(diaper_beer) == 1
    row = diaper_beer.iloc[0]
    assert row["support"] == pytest.approx(0.6)
    assert row["confidence"] == pytest.approx(0.75)
    assert row["lift"] == pytest.approx(1.25)

    # Results are sorted by decreasing confidence.
    assert rules["confidence"].is_monotonic_decreasing


def test_generate_rules_accepts_dataframe_and_filters_by_lift():
    frequent = ar.apriori(TOY, min_support=0.2)
    frame = ar.itemsets_to_dataframe(frequent)

    rules = ar.generate_rules(frame, min_confidence=0.5, min_lift=1.2)
    assert not rules.empty
    assert (rules["lift"] >= 1.2 - 1e-12).all()

    # association_rules is an alias with identical behaviour.
    alias_rules = ar.association_rules(frame, min_confidence=0.5, min_lift=1.2)
    pd.testing.assert_frame_equal(rules, alias_rules)


def test_generate_rules_perfect_rule_has_infinite_conviction():
    frequent = ar.apriori(TOY, min_support=0.2)
    rules = ar.generate_rules(frequent, min_confidence=1.0)

    beer_to_diaper = rules[
        (rules["antecedents"] == frozenset({"Beer"}))
        & (rules["consequents"] == frozenset({"Diaper"}))
    ]
    assert len(beer_to_diaper) == 1
    assert np.isinf(beer_to_diaper.iloc[0]["conviction"])


def test_generate_rules_validation_and_empty_input():
    frequent = ar.apriori(TOY, min_support=0.2)
    with pytest.raises(ValueError, match="min_confidence"):
        ar.generate_rules(frequent, min_confidence=-0.1)
    with pytest.raises(ValueError, match="min_confidence"):
        ar.generate_rules(frequent, min_confidence=1.1)
    with pytest.raises(ValueError, match="min_lift"):
        ar.generate_rules(frequent, min_lift=np.nan)
    with pytest.raises(TypeError):
        ar.generate_rules([frozenset({"Bread", "Milk"})])

    empty_rules = ar.generate_rules({}, min_confidence=0.5)
    assert empty_rules.empty
    assert list(empty_rules.columns) == ar.RULES_COLUMNS

    # A mapping with only singletons produces no rules.
    singleton_rules = ar.generate_rules(ar.apriori(TOY, min_support=1.0))
    assert singleton_rules.empty


# ---------------------------------------------------------------------------
# Edge cases for the miners
# ---------------------------------------------------------------------------
def test_empty_transaction_dataset():
    for model_class in (ar.Apriori, ar.FPGrowth):
        model = model_class(min_support=0.1).fit([])
        assert model.n_transactions_ == 0
        assert model.frequent_itemsets_ == {}
        assert model.support_counts_ == {}
        if model_class is ar.Apriori:
            assert model.level_stats_ == []

    assert ar.apriori([], min_support=0.1) == {}
    assert ar.fpgrowth([], min_support=0.1) == {}
    assert ar.generate_rules({}).empty


def test_single_transaction_and_one_item_datasets():
    transactions = [["Bread", "Milk", "Eggs"]]

    for model_class in (ar.Apriori, ar.FPGrowth):
        model = model_class(min_support=0.5).fit(transactions)
        expected = {
            frozenset({"Bread"}),
            frozenset({"Milk"}),
            frozenset({"Eggs"}),
            frozenset({"Bread", "Milk"}),
            frozenset({"Bread", "Eggs"}),
            frozenset({"Milk", "Eggs"}),
            frozenset({"Bread", "Milk", "Eggs"}),
        }
        assert set(model.frequent_itemsets_) == expected
        assert all(value == pytest.approx(1.0) for value in model.frequent_itemsets_.values())

    one_item = ar.Apriori(min_support=1.0).fit([["Bread"], ["Bread"]])
    assert one_item.frequent_itemsets_ == {frozenset({"Bread"}): 1.0}


def test_invalid_miner_parameters():
    for model_class in (ar.Apriori, ar.FPGrowth):
        for bad_support in (-0.1, 0.0, 1.1, np.nan, np.inf):
            with pytest.raises(ValueError, match="min_support"):
                model_class(min_support=bad_support)
        with pytest.raises(ValueError, match="max_len"):
            model_class(min_support=0.1, max_len=0)
        with pytest.raises(ValueError, match="max_len"):
            model_class(min_support=0.1, max_len=1.5)


def test_itemsets_to_dataframe():
    mapping = {
        frozenset({"Bread"}): 0.8,
        frozenset({"Bread", "Milk"}): 0.6,
        frozenset({"Eggs"}): 0.2,
    }
    frame = ar.itemsets_to_dataframe(mapping)

    assert list(frame.columns) == ["support", "itemsets"]
    assert frame["support"].tolist() == [0.8, 0.2, 0.6]
    # Sorted by itemset size first, then alphabetically.
    assert frame["itemsets"].tolist() == [
        frozenset({"Bread"}),
        frozenset({"Eggs"}),
        frozenset({"Bread", "Milk"}),
    ]


# ---------------------------------------------------------------------------
# Synthetic warung data
# ---------------------------------------------------------------------------
def test_generate_warung_transactions_is_reproducible_and_valid():
    first = ar.generate_warung_transactions(300, random_state=42)
    second = ar.generate_warung_transactions(300, random_state=42)
    other = ar.generate_warung_transactions(300, random_state=7)

    assert first == second
    assert first != other
    assert len(first) == 300
    assert all(isinstance(basket, list) and basket for basket in first)
    assert all(
        isinstance(item, str) and item in ar.WARUNG_PRODUCTS
        for basket in first
        for item in basket
    )
    # No duplicate item inside a basket.
    assert all(len(basket) == len(set(basket)) for basket in first)


def test_generate_warung_transactions_has_planted_associations():
    transactions = ar.generate_warung_transactions(500, random_state=42)

    for left, right in ar.PLANTED_ASSOCIATIONS:
        pair_support = ar.support(transactions, {left, right})
        assert pair_support > 0.05
        assert ar.lift(transactions, left, right) > 1.0
        assert ar.confidence(transactions, left, right) > \
            ar.support(transactions, right)

    matrix = ar.one_hot_encode(transactions)
    assert matrix.shape == (500, len(ar.WARUNG_PRODUCTS))
    assert matrix.to_numpy().sum() == sum(len(basket) for basket in transactions)


def test_generate_warung_transactions_edge_cases():
    assert ar.generate_warung_transactions(0, random_state=1) == []
    with pytest.raises(ValueError, match="non-negative"):
        ar.generate_warung_transactions(-1, random_state=1)
    with pytest.raises(ValueError, match="integer"):
        ar.generate_warung_transactions(3.5, random_state=1)


def test_generate_warung_transactions_end_to_end_mining():
    transactions = ar.generate_warung_transactions(400, random_state=42)
    frequent = ar.apriori(transactions, min_support=0.03)
    rules = ar.generate_rules(frequent, min_confidence=0.45)

    assert rules.shape[0] > 0
    assert (rules["confidence"] >= 0.45 - 1e-12).all()

    # All four planted pairs should appear as single-item antecedent rules.
    found = {
        (next(iter(row["antecedents"])), next(iter(row["consequents"])))
        for _, row in rules.iterrows()
        if len(row["antecedents"]) == 1 and len(row["consequents"]) == 1
    }
    for pair in ar.PLANTED_ASSOCIATIONS:
        assert pair in found


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
