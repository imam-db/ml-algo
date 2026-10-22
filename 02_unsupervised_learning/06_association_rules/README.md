# Association Rule Mining (Apriori & FP-Growth)

**🇬🇧 English | [🇮🇩 Bahasa Indonesia](README.id.md)**

**🏠 [Back to Main](../../README.md)** | **📁 [Unsupervised Learning](../README.md)** | **⬅️ [Previous: PCA](../04_pca/README.md)** | **🧠 [Related: t-SNE](../05_tsne/)** *(planned)*

## 🗺️ Quick Access

| File | Description | Quick Run |
|------|-------------|-----------|
| [`implementation.py`](./implementation.py) | From-scratch Apriori, FP-Growth, metrics and warung data generator | `python3 implementation.py` |
| [`sklearn_example.py`](./sklearn_example.py) | Compares our results with `mlxtend` (optional dependency) | `python3 sklearn_example.py` |
| [`exercise.ipynb`](./exercise.ipynb) | Guided exercises with TODOs and hints | `jupyter lab exercise.ipynb` |

## 📚 Table of Contents

- [Theory & Concepts](#-theory--concepts)
- [Mathematical Foundation](#-mathematical-foundation)
- [The Apriori Algorithm](#-the-apriori-algorithm)
- [The FP-Growth Algorithm](#-the-fp-growth-algorithm)
- [Apriori vs FP-Growth](#-apriori-vs-fp-growth)
- [The Warung Dataset](#-the-warung-dataset)
- [Metrics in Practice](#-metrics-in-practice)
- [Implementation Details](#-implementation-details)
- [Advantages & Disadvantages](#-advantages--disadvantages)
- [When to Use](#-when-to-use)
- [Common Pitfalls & Best Practices](#-common-pitfalls--best-practices)
- [Files in This Directory](#-files-in-this-directory)
- [Quick Start](#-quick-start)
- [Exercises](#-exercises)
- [Related Algorithms](#-related-algorithms)
- [References](#-references)

## 🧠 Theory & Concepts

### What is Association Rule Mining?

Association rule mining finds items that occur together in transactional data.
A **transaction** is one basket, receipt, click session, or shopping cart. An
**itemset** is any set of items. The goal is to discover rules of the form:

```
X -> Y
```

where `X` (the **antecedent**) and `Y` (the **consequent**) are disjoint
non-empty itemsets. A classic example is:

```
{Indomie Goreng} -> {Telur Ayam}
```

If a customer buys Indomie, how likely are they to buy eggs as well? This
question is useful for shelf placement, bundling, cross-selling, promotion
design, and recommendation systems.

Unlike supervised learning, association rule mining has **no target label**.
It is an unsupervised pattern-discovery method: it searches for regularities in
co-occurrence rather than predicting a predefined outcome.

### Transactions, Itemsets and Rules

| Term | Meaning | Example |
|------|---------|---------|
| Item | One product or event | `Indomie Goreng` |
| Transaction | One basket of items | `{Indomie, Telur, Rokok}` |
| Itemset | Any set of items | `{Kopi Sachet, Gula Pasir}` |
| k-itemset | Itemset with `k` items | `{Roti, Susu}` is a 2-itemset |
| Rule | Implication between itemsets | `{Rokok} -> {Korek Api}` |
| Support | How often an itemset appears | 20% of baskets contain both |
| Confidence | How often `Y` appears when `X` appears | 65% of rokok baskets have korek |

### Why Support and Confidence Are Not Enough

Confidence alone can be misleading. A rule `{Bread} -> {Milk}` may have high
confidence simply because **milk is in almost every basket**. The bread adds
little information. **Lift** corrects for this by comparing the observed
co-occurrence with what independence would predict:

```
lift = confidence / support(Y)
     = support(X ∪ Y) / (support(X) * support(Y))
```

- `lift > 1`: the items are positively associated (bought together more than
  chance).
- `lift = 1`: independent.
- `lift < 1`: negatively associated (substitutes or mutually exclusive).

Two further metrics are useful:

- **Leverage** measures the absolute difference from independence:
  `support(X ∪ Y) - support(X) * support(Y)`.
- **Conviction** measures how strongly the rule is implied:
  `(1 - support(Y)) / (1 - confidence)`. It is infinite for a perfect rule.

### Frequent Itemsets Are the Foundation

Rather than testing every possible rule directly, association-rule mining works
in two stages:

1. **Frequent-itemset mining**: find all itemsets whose support is at least
   `min_support`.
2. **Rule generation**: split each frequent itemset into antecedent/consequent
   pairs and keep the rules whose confidence is at least `min_confidence`.

This two-stage design is efficient because the number of frequent itemsets is
usually much smaller than the number of possible rules.

## 📐 Mathematical Foundation

Let `D` be a dataset of `N` transactions, and let `X`, `Y` be itemsets.

### Support

```
support(X) = |{t in D : X ⊆ t}| / N
```

Support is a fraction between 0 and 1. Some tools accept an absolute count
instead; this module uses a **relative** threshold in `(0, 1]`.

### Confidence

```
confidence(X -> Y) = support(X ∪ Y) / support(X)
```

Confidence is undefined when `support(X) = 0`. It is asymmetric: generally
`confidence(X -> Y) != confidence(Y -> X)`.

### Lift

```
lift(X -> Y) = confidence(X -> Y) / support(Y)
             = support(X ∪ Y) / (support(X) * support(Y))
```

Lift is symmetric in `X` and `Y` even though confidence is not.

### Leverage

```
leverage(X -> Y) = support(X ∪ Y) - support(X) * support(Y)
```

Leverage is also symmetric. A positive value means the items co-occur more often
than independence predicts. It is often easier to interpret than lift when one
item is very rare, because it is not inflated by a tiny denominator.

### Conviction

```
conviction(X -> Y) = (1 - support(Y)) / (1 - confidence(X -> Y))
```

- `conviction = 1`: the rule gives no information beyond the base rate of `Y`.
- `conviction > 1`: the rule is informative.
- `conviction = ∞`: confidence is 1 (a perfect rule), provided `support(Y) < 1`.
- If both confidence and `support(Y)` are 1, the ratio is `0/0`; this
  implementation defines conviction as `1.0` in that degenerate case.

### Anti-Monotonicity of Support

The key property behind Apriori is:

> If an itemset is frequent, every subset of it is also frequent. Equivalently,
> if an itemset is infrequent, every superset of it is infrequent.

Proof sketch: if `X ⊆ Z`, then every transaction containing `Z` also contains
`X`. Therefore `support(X) >= support(Z)`. If `support(X) < min_support`, then
`support(Z)` must also be below the threshold.

This property lets Apriori discard an enormous number of candidates without
ever counting them.

## ⚙️ The Apriori Algorithm

Apriori (Agrawal & Srikant, 1994) mines frequent itemsets **level by level**.

### Step 1: Frequent 1-itemsets

Count every individual item and keep those with support at least
`min_support`. Call this set `L1`.

### Step 2: Join Step

To build candidates of size `k`, join pairs of frequent `(k-1)`-itemsets that
share the same first `k-2` items. With items sorted inside each itemset, if

```
A = {a1, ..., a(k-2), a(k-1)}
B = {a1, ..., a(k-2), b(k-1)}
```

then the join produces:

```
C = {a1, ..., a(k-2), a(k-1), b(k-1)}
```

The shared prefix keeps the number of candidate pairs manageable.

### Step 3: Anti-Monotone Pruning

Before counting a candidate, check every `(k-1)`-subset. If any subset is not in
`L(k-1)`, the candidate cannot be frequent and is removed immediately. This is
the **subset check** that gives Apriori its name.

### Step 4: Count and Filter

Scan the transactions once and count how many contain each surviving candidate.
Keep candidates whose support is at least `min_support`. Repeat until no new
frequent itemsets are found or `max_len` is reached.

### Worked Example

Consider five toy baskets:

```
1. {Bread, Milk}
2. {Bread, Diaper, Beer, Eggs}
3. {Milk, Diaper, Beer, Cola}
4. {Bread, Milk, Diaper, Beer}
5. {Bread, Milk, Diaper, Cola}
```

At `min_support = 0.2` (at least one basket):

| Itemset | Count | Support |
|---------|-------|---------|
| `{Bread}` | 4 | 0.8 |
| `{Milk}` | 4 | 0.8 |
| `{Diaper}` | 4 | 0.8 |
| `{Beer}` | 3 | 0.6 |
| `{Eggs}` | 1 | 0.2 |
| `{Cola}` | 2 | 0.4 |
| `{Bread, Milk}` | 3 | 0.6 |
| `{Diaper, Beer}` | 3 | 0.6 |

The rule `{Diaper} -> {Beer}` has:

```
support    = 0.6
confidence = 0.6 / 0.8 = 0.75
lift       = 0.75 / 0.6 = 1.25
leverage   = 0.6 - 0.8 * 0.6 = 0.12
conviction = (1 - 0.6) / (1 - 0.75) = 1.6
```

Because every basket containing beer also contains diaper, the reverse rule
`{Beer} -> {Diaper}` has confidence `1.0` and infinite conviction.

### Complexity

The worst case is exponential in the number of distinct items, because there are
`2^m - 1` possible itemsets. In practice, anti-monotone pruning and a sensible
`min_support` keep the search small. Candidate counting dominates the runtime:
each level scans the transactions and checks the surviving candidates.

## 🌳 The FP-Growth Algorithm

FP-Growth (Han, Pei & Yin, 2000) avoids candidate generation entirely. It
compresses the database into a tree and mines that tree recursively.

### Step 1: Count Items and Build the Header Table

Scan the database once and count item frequencies. Discard items below
`min_support`. Sort the remaining items by descending frequency and build a
header table with a link to every node for that item.

### Step 2: Build the FP-Tree

For each transaction:

1. keep only frequent items,
2. sort them by the header-table order,
3. insert them into the tree, sharing prefixes with existing paths,
4. increment the count of every node on the path.

The result is a compact prefix tree. A transaction that shares a prefix with an
existing transaction does not need a new branch.

### Step 3: Mine Conditional Pattern Bases

For each item in the header table, walk its node links up to the root. Each path
is a **conditional pattern base**: the set of items that co-occur with the
suffix item, weighted by the node count.

### Step 4: Build Conditional FP-Trees and Recurse

Count the items in the pattern base, discard those below `min_support`, and
build a smaller **conditional FP-tree**. Recurse with the suffix item added to
the current prefix. Every recursive call produces a frequent itemset.

Because FP-Growth never enumerates candidates, it often scales better than
Apriori on dense datasets. The trade-off is a more complex implementation and
extra memory for the tree and header links.

## ⚖️ Apriori vs FP-Growth

| Aspect | Apriori | FP-Growth |
|--------|---------|-----------|
| Main idea | Level-wise candidate generation | Compressed FP-tree, no candidates |
| Database scans | One per level (multiple) | Two (build tree, then mine) |
| Candidate generation | Yes (join + prune) | No |
| Memory | Candidate sets can be large | Tree + header links |
| Best for | Sparse data, short itemsets, teaching | Dense data, many repeated patterns |
| Implementation complexity | Moderate | Higher |
| Result | All frequent itemsets | All frequent itemsets |

Both algorithms return exactly the same frequent itemsets and supports for the
same `min_support`. The test suite checks this equivalence, including on the
synthetic warung data.

## 🏪 The Warung Dataset

`generate_warung_transactions(n_transactions=1000, random_state=42)` creates
reproducible synthetic baskets from an Indonesian warung/minimarket:

- **18 products** such as Indomie Goreng, telur, kopi sachet, gula pasir, rokok,
  korek api, roti tawar, susu UHT, air mineral, sabun, and shampoo sachet.
- Each basket starts with a random set of 1–3 weighted products.
- With probability 0.70, one **planted association** is added:

| Planted pair | Story |
|--------------|-------|
| `Indomie Goreng` + `Telur Ayam` | The classic fried-noodle meal |
| `Kopi Sachet` + `Gula Pasir` | Sweet coffee at home |
| `Rokok` + `Korek Api` | Cigarettes need a lighter |
| `Roti Tawar` + `Susu UHT` | Breakfast combination |

The generator keeps at most one planted pair per basket so that cross-pair
combinations do not dominate the mined rules. Typical results on 800 baskets
with `min_support = 0.02`:

```
Apriori  : 137 frequent itemsets
FP-Growth: 137 frequent itemsets
Identical: True

{Indomie Goreng} -> {Telur Ayam}  conf ≈ 0.62  lift ≈ 2.0
{Kopi Sachet}    -> {Gula Pasir}   conf ≈ 0.66  lift ≈ 2.5
{Rokok}          -> {Korek Api}    conf ≈ 0.55  lift ≈ 2.6
{Roti Tawar}     -> {Susu UHT}     conf ≈ 0.67  lift ≈ 2.4
```

The exact numbers depend on `n_transactions` and `random_state`, but the planted
pairs consistently have lift well above 1.

## 📊 Metrics in Practice

### Choosing `min_support`

`min_support` controls the size of the search:

```
Too low  -> many itemsets, slow runtime, noisy rules
Too high -> few itemsets, misses useful associations
```

A practical workflow:

1. Encode baskets and inspect item frequencies.
2. Start with a support threshold that keeps itemsets in the hundreds or low
   thousands.
3. Mine itemsets with Apriori or FP-Growth.
4. Generate rules with a meaningful `min_confidence`.
5. Rank by lift or leverage, then check support so that rules are not based on
   only a handful of baskets.
6. Validate the best rules on a hold-out period or with a business experiment.

### Reading a Rule Table

| Metric | High value means | Watch out for |
|--------|------------------|---------------|
| Support | The pattern is common | High support can be trivial (milk in every basket) |
| Confidence | `Y` usually follows `X` | Can be high when `Y` is simply very popular |
| Lift | Strong positive association | Very rare items can produce unstable, huge lift |
| Leverage | Large absolute effect | Harder to compare across datasets with different sizes |
| Conviction | Strong implication | Infinite for perfect rules; not comparable across all rules |

### A Rule Is Not Causation

`{Rokok} -> {Korek Api}` does not mean cigarettes *cause* lighter purchases. It
means the two items co-occur. Use domain knowledge, experiments, and temporal
validation before acting on a rule.

## 🛠️ Implementation Details

### Public API

```python
from implementation import (
    encode_transactions, one_hot_encode,
    support, confidence, lift, leverage, conviction, rule_metrics,
    Apriori, apriori,
    FPGrowth, fpgrowth,
    generate_rules, association_rules,
    generate_warung_transactions,
)
```

### Mining Frequent Itemsets

```python
transactions = generate_warung_transactions(800, random_state=42)

model = Apriori(min_support=0.03, max_len=None).fit(transactions)
print(len(model.frequent_itemsets_))
print(model.to_dataframe().head())

# Function form: returns {frozenset(items): support}
frequent = apriori(transactions, min_support=0.03)

# FP-Growth returns exactly the same mapping
frequent_fp = fpgrowth(transactions, min_support=0.03)
assert frequent == frequent_fp
```

### Generating Rules

```python
rules = generate_rules(
    model.frequent_itemsets_,
    min_confidence=0.50,
    min_lift=1.0,
)

print(rules[["antecedents", "consequents", "support",
             "confidence", "lift"]].head())
```

The rule DataFrame contains:

```
antecedents, consequents,
antecedent_support, consequent_support, support,
confidence, lift, leverage, conviction
```

### Parameters

| Parameter | Meaning | Typical values |
|-----------|---------|----------------|
| `min_support` | Minimum relative support | `0.01`–`0.10` |
| `min_confidence` | Minimum rule confidence | `0.30`–`0.80` |
| `min_lift` | Minimum lift | `1.0`–`2.0` |
| `max_len` | Maximum itemset size | `2`–`4` for interpretability |

### Edge Cases

- Empty transaction lists return no itemsets and no rules.
- Empty baskets are preserved so the support denominator is correct.
- Duplicate items inside one basket are removed by `encode_transactions`.
- `min_support` must be in `(0, 1]`; `0` would make every possible subset
  "frequent" and is intentionally rejected.
- Confidence with a zero-support antecedent raises a clear `ValueError`.
- Perfect rules have infinite conviction, except when the consequent also has
  support 1.

## ✅ Advantages & Disadvantages

### ✅ Advantages

- Easy to interpret: rules are expressed in terms of real products.
- No labels required; works directly on transaction logs.
- Apriori is simple to implement and to explain.
- FP-Growth is fast on dense datasets and avoids candidate explosion.
- Metrics such as lift and leverage separate genuine associations from popular
  items.
- Naturally produces actionable outputs for retail, web analytics, and
  recommendation systems.

### ❌ Disadvantages

- Frequent-itemset mining can be expensive: the number of itemsets grows
  exponentially with the number of items.
- Choosing `min_support` and `min_confidence` requires experimentation.
- Rare-item rules can have unstable or misleadingly large lift.
- Classic algorithms use only presence/absence, not quantity or price.
- Rules are correlational, not causal.
- Temporal order is ignored unless transactions are modeled explicitly over
  time.
- Very large catalogs may need constrained mining (for example, only pairs, or
  only rules containing a target item).

## 🎯 When to Use

### ✅ Use association rules when

- You have transactional data: receipts, baskets, clicks, or event logs.
- You want human-readable "if X then Y" patterns.
- You need to support shelf placement, bundling, or cross-selling decisions.
- You want to explore co-occurrence before building a recommender.
- The catalog is moderate, or you can restrict `max_len` / support.

### ❌ Consider alternatives when

- You need to predict a specific target: use classification or regression.
- You need a personalized ranking: use collaborative filtering or a recommender
  system.
- The data is dense and continuous rather than transactional: use clustering or
  correlation analysis.
- You need causal effects: use controlled experiments or causal inference.
- The item catalog is enormous and the transaction volume is huge: consider
  specialized frequent-pattern libraries, sampling, or constrained mining.

## 💡 Common Pitfalls & Best Practices

1. **Do not use confidence alone.** Always inspect lift or leverage.
2. **Check support.** A rule with lift 20 based on three baskets is not
   actionable.
3. **Beware popular consequents.** Milk, air mineral, and cigarettes can make
   confidence look impressive.
4. **Use `max_len` for readability.** Pair rules are usually easier to act on
   than six-item rules.
5. **Validate out of time.** A rule mined on one month may not hold in the next.
6. **Translate rules into actions.** "Put korek api near rokok" is more useful
   than a table of numbers.
7. **Watch the threshold boundary.** Support is inclusive: exactly
   `min_support` counts as frequent.
8. **Remember the anti-monotone property.** It is the reason pruning is safe and
   the reason a low threshold can explode.

## 📁 Files in This Directory

- **`README.md`** (this file) — theory and concepts in English
- **`README.id.md`** — dokumentasi lengkap dalam Bahasa Indonesia
- **`implementation.py`** — transaction encoding, support/confidence/lift/
  leverage/conviction, Apriori, FP-Growth, rule generation, synthetic warung
  data, and a `__main__` demo
- **`sklearn_example.py`** — compares our results with `mlxtend`; if `mlxtend`
  is not installed it prints a message and exits successfully
- **`exercise.ipynb`** — hands-on exercises with TODOs and hints

## 🚀 Quick Start

Run the from-scratch demo:

```bash
python3 02_unsupervised_learning/06_association_rules/implementation.py
```

Run the optional `mlxtend` comparison:

```bash
python3 02_unsupervised_learning/06_association_rules/sklearn_example.py
```

Run the tests:

```bash
pytest tests/test_association_rules.py -v
```

Minimal usage:

```python
from implementation import Apriori, generate_rules, generate_warung_transactions

transactions = generate_warung_transactions(800, random_state=42)
model = Apriori(min_support=0.03).fit(transactions)
rules = generate_rules(model.frequent_itemsets_, min_confidence=0.50)

print(rules.sort_values("lift", ascending=False).head(10))
```

## 🎓 Exercises

The notebook [`exercise.ipynb`](./exercise.ipynb) walks through:

1. generating and inspecting warung baskets,
2. encoding transactions and counting item frequencies,
3. computing the five metrics by hand,
4. mining itemsets with Apriori,
5. verifying that FP-Growth returns the same result,
6. generating and filtering rules,
7. recovering the planted associations,
8. measuring the effect of `min_support`,
9. observing anti-monotone pruning in action.

Every cell runs top-to-bottom; TODO prompts and hints encourage you to change
thresholds and observe the result.

## 🔗 Related Algorithms

- **K-Means** ([01_kmeans](../01_kmeans/)): groups similar transactions or
  customers; association rules describe co-occurrence inside baskets.
- **PCA** ([04_pca](../04_pca/)): reduces feature dimensions; association rules
  work on categorical presence/absence.
- **DBSCAN** ([02_dbscan](../02_dbscan/)) and **Hierarchical Clustering**
  ([03_hierarchical](../03_hierarchical/)): alternative unsupervised grouping
  methods.
- **Naive Bayes** ([06_naive_bayes](../../01_supervised_learning/06_naive_bayes/)):
  probabilistic classification; can be combined with basket features.
- **Recommender systems** ([06_playground](../../06_playground/)): usually use
  collaborative filtering, but association rules are a strong, explainable
  baseline.

## 📚 References

- Agrawal, R., & Srikant, R. (1994). *Fast Algorithms for Mining Association
  Rules*. Proceedings of VLDB.
- Han, J., Pei, J., & Yin, Y. (2000). *Mining Frequent Patterns without
  Candidate Generation*. SIGMOD.
- [mlxtend frequent pattern mining documentation](https://rasbt.github.io/mlxtend/user_guide/frequent_patterns/)
- [Association rule learning (Wikipedia)](https://en.wikipedia.org/wiki/Association_rule_learning)
- [Apriori algorithm (Wikipedia)](https://en.wikipedia.org/wiki/Apriori_algorithm)
- [FP-Growth (Wikipedia)](https://en.wikipedia.org/wiki/FP-growth_algorithm)

---

**Next Steps**:
- Read `implementation.py` to see the join step, subset pruning, and FP-tree
  recursion.
- Run `sklearn_example.py` to compare with `mlxtend` (install it first with
  `pip install mlxtend`).
- Complete `exercise.ipynb` and try a `max_len=2` analysis for shelf placement.
- Apply the same workflow to your own transaction data.
