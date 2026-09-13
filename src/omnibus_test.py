"""
Omnibus significance test for the matched-control comparison.

Reproduces every statistic reported in Section V-A of the paper from the
per-seed paired differences saved in results/results_1c_hard.pkl. No GPU, no
dataset and no model loading required -- it runs in about a second.

The four per-K tests share seeds, so they are not mutually independent and two
of them are individually inconclusive at S = 8. The omnibus forms below
aggregate over K and are the operative statement.

    python src/omnibus_test.py
"""

import pickle
from pathlib import Path

import numpy as np
from scipy import stats

RESULTS = Path(__file__).resolve().parents[1] / "results" / "results_1c_hard.pkl"


def load_paired_differences(path=RESULTS):
    """Return (K values, array of shape (n_K, n_seeds) of quantum - classical)."""
    with open(path, "rb") as fh:
        data = pickle.load(fh)
    Ks = data["Ks"]
    return Ks, np.array([data["paired_qc"][k] for k in Ks])


def per_budget_tests(Ks, D):
    """Paired t-test at each K -- the four tests reported in the text."""
    return [
        (k, D[i].mean(), *stats.ttest_1samp(D[i], 0))
        for i, k in enumerate(Ks)
    ]


def omnibus(D):
    """Three aggregate tests over the whole design."""
    n_K, S = D.shape

    # 1. Average within seed, then one paired t-test on S independent units.
    #    This is eq. (8) applied to the seed-level means -- the headline form.
    per_seed = D.mean(axis=0)
    t, p_two = stats.ttest_1samp(per_seed, 0)

    # 2. Distribution-free counterpart, in case eight points is too few to
    #    trust normality.
    signed_rank = stats.wilcoxon(per_seed, alternative="less")

    # 3. Hotelling T^2 on the full 4-vector of per-K differences. Valid only
    #    because S > n_K; reported for completeness, not relied on.
    mean_vec = D.mean(axis=1)
    T2 = S * mean_vec @ np.linalg.inv(np.cov(D)) @ mean_vec
    F = (S - n_K) / (n_K * (S - 1)) * T2
    p_hotelling = stats.f.sf(F, n_K, S - n_K)

    # 4. Sign test over every seed x K cell, ignoring magnitudes entirely.
    n_neg = int((D < 0).sum())
    p_sign = stats.binomtest(n_neg, D.size, 0.5, alternative="greater").pvalue

    return {
        "seed_mean": per_seed.mean(),
        "t": t,
        "df": S - 1,
        "p_two_sided": p_two,
        "p_one_sided": p_two / 2 if t < 0 else 1 - p_two / 2,
        "wilcoxon_p": signed_rank.pvalue,
        "hotelling_F": F,
        "hotelling_p": p_hotelling,
        "n_negative": n_neg,
        "n_cells": D.size,
        "p_sign": p_sign,
    }


if __name__ == "__main__":
    Ks, D = load_paired_differences()
    rows = per_budget_tests(Ks, D)
    o = omnibus(D)

    print(
        "per-budget paired differences (quantum - classical), S = %d seeds:\n"
        % D.shape[1]
        + "\n".join(
            "  K=%2d  d=%+.4f  t=%+.3f  p=%.4f" % (k, d, t, p) for k, d, t, p in rows
        )
    )
    print(
        "\nomnibus: seed-level mean d=%+.4f, t(%d)=%+.3f, p=%.4f (two-sided); "
        "signed-rank p=%.4f; Hotelling F(%d,%d)=%.2f, p=%.4f; "
        "sign test %d/%d negative, p=%.4f"
        % (
            o["seed_mean"], o["df"], o["t"], o["p_two_sided"],
            o["wilcoxon_p"], D.shape[0], D.shape[1] - D.shape[0],
            o["hotelling_F"], o["hotelling_p"],
            o["n_negative"], o["n_cells"], o["p_sign"],
        )
    )
