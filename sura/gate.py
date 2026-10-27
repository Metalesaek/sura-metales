"""Statistical acceptance tests: does a candidate adapter harm old knowledge?

Both tests compare the candidate against the ORIGINAL frozen base (cumulative drift),
per regression-suite item d_i (see runner.harm_d; default: loss of preference margin
for the true answer over the counterfactual).

  H0: mean(d) = mu0          (normal drift of clean learning, calibrated) -> commit
  H1: mean(d) = mu0 + delta  (harmful)                                   -> rollback

ttest  one-sided paired t-test of mean(d) > delta on the whole suite
sprt   Wald's sequential probability ratio test on suite mini-batches; stops as soon
       as the evidence is decisive, so most decisions score only part of the suite
"""
from __future__ import annotations

import math
from typing import Callable

import torch
from scipy import stats


def ttest_harm(diff: torch.Tensor, delta: float, alpha: float) -> tuple[bool, dict]:
    n = diff.numel()
    mean, sd = diff.mean().item(), diff.std(unbiased=True).item()
    t = (mean - delta) / (sd / math.sqrt(n) + 1e-12)
    p = float(stats.t.sf(t, df=n - 1))
    return p < alpha, dict(test="ttest", mean_diff=mean, p_value=p, items_scored=n)


def sprt_harm(score_batch: Callable[[int, int], torch.Tensor], n_items: int, delta: float,
              alpha: float, beta: float, batch: int, mu0: float = 0.0) -> tuple[bool, dict]:
    """`score_batch(i, j)` returns d for suite items i..j-1 (scored lazily).
    H0: mean(d) = mu0 (normal drift of clean learning)  vs  H1: mean(d) = mu0 + delta."""
    upper = math.log((1 - beta) / alpha)   # cross -> accept H1 (harm)
    lower = math.log(beta / (1 - alpha))   # cross -> accept H0 (harmless)
    seen: list[torch.Tensor] = []
    llr, decided, harm = 0.0, False, False
    for i in range(0, n_items, batch):
        d = score_batch(i, min(i + batch, n_items))
        seen.append(d)
        allv = torch.cat(seen)
        var = max(allv.var(unbiased=True).item() if allv.numel() > 1 else 1.0, 1e-6)
        # Gaussian LLR for mean delta vs 0, plug-in variance, recomputed on all items so far
        llr = (delta / var) * (allv.sum().item() - allv.numel() * (mu0 + delta / 2))
        if llr >= upper:
            decided, harm = True, True
            break
        if llr <= lower:
            decided, harm = True, False
            break
    allv = torch.cat(seen)
    if not decided:
        harm = llr > 0
    return harm, dict(test="sprt", mean_diff=allv.mean().item(), llr=llr, decided=decided,
                      items_scored=allv.numel())
