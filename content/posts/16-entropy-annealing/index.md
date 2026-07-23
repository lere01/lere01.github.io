---
author: Faith O. Oyedemi
title: "Entropy Annealing: Keeping a Wavefunction Curious"
date: 2026-02-11
draft: true
tags:
- entropy annealing
- variational Monte Carlo
- optimization
---

{{< katex >}}

When a probability model becomes too confident too early, asking it to optimize harder may only deepen the mistake. My most effective response to amplitude collapse has been to reward uncertainty temporarily, then remove that reward gradually.

I think of entropy annealing as keeping the wavefunction curious long enough to discover configurations it would otherwise erase.

## Energy alone can favour premature concentration

Variational Monte Carlo minimizes

$$
E_\theta= \mathbb E_{\sigma\sim p_\theta} [E_{\mathrm{loc}}(\sigma)].
$$

Early in training, both the local energies and their gradients are noisy. If a small set of configurations appears favourable, the autoregressive distribution can concentrate rapidly. Once that happens, alternative configurations contribute little to subsequent updates.

We add a temporary entropy term through a free-energy-like objective

$$
F_T=E_\theta-TS[p_\theta],
$$

where

$$
S[p_\theta]=-\sum_\sigma p_\theta(\sigma)\log p_\theta(\sigma)
$$

is the Shannon entropy and \\(T\ge0\\) is an artificial training temperature.

At \\(T>0\\), a broader distribution is rewarded. As \\(T\to0\\), the objective returns to the physical variational energy.

The temperature is an optimization device. It is not the physical temperature of the quantum system.

## A sample-level form

Because

$$
-S[p]=\mathbb E_p[\log p(\sigma)],
$$

we can estimate the regularized objective from the same exact autoregressive samples used for energy. A centred local form is

$$
F_{\mathrm{loc}}(\sigma) = E_{\mathrm{loc}}(\sigma) + T\left( \log p_\theta(\sigma) -\langle\log p_\theta\rangle \right).
$$

Centering does not change the expected gradient in the intended estimator, but it reduces the influence of an arbitrary offset and mirrors the covariance structure used in VMC gradients.

## Why anneal instead of keeping entropy forever?

The ground state is determined by \\(H\\), not by the Shannon entropy of one chosen basis representation. A permanent \\(T>0\\) solves a different optimization problem.

An annealing schedule might begin at \\(T_0\\), remain warm for a period, then decrease to zero:

$$
T(t)= \begin{cases} T_0, & t<t_{\mathrm{hold}},\\ T_0\,g(t), & t_{\mathrm{hold}}\le t<t_{\mathrm{end}},\\ 0, & t\ge t_{\mathrm{end}}. \end{cases}
$$

The function \\(g(t)\\) may be linear, cosine, or exponential. What matters scientifically is that the schedule is recorded and that the final optimization actually targets energy.

## What happened in the J1-J2 experiments

The clearest benefit appeared where unannealed models collapsed onto classical configurations.

For \\(6\times6\\) cells in the Néel-side study, unannealed DPMixer runs froze near classical energies as their variance approached zero. With the annealed recipe, corresponding production runs reached approximately 0.17% and 0.28% relative error.

At \\(J_2=0.8\\), a broader architecture grid moved from classical or 19-22% plateaus into roughly 0.5-1% error. On \\(4\times4\\), several gauged, annealed runs reached errors of order \\(0.01\%-0.07\%\\) across five seeds in the selected campaign.

These numbers describe the archived configurations and references used in this project. They should not be generalized to every architecture or schedule without replication.

## What entropy did not solve

The most important negative result is that annealing did not reliably choose the correct prior-free sign basin.

It protected amplitude support and cured deterministic collapse. Yet prior-free sign learning remained bimodal across seeds: some runs found near-exact signs, while others retained partial sign agreement. Changing the annealing window did not reliably steer that choice.

This separates two problems:

1. **Amplitude collapse:** the model loses configurations needed for quantum correlations.
2. **Sign-basin selection:** the model settles on an incorrect interference pattern.

Entropy directly addresses the first. It may create better conditions for the second, but it does not determine the answer.

## Architecture and learning rate still matter

An annealed Transformer evaluated far from its preferred sample and learning-rate regime can still fail badly. In the internal campaign, the Transformer at 2,000 samples followed classical orbits across the \\(J_2\\) grid, whereas its native 200-sample setting produced sub-percent results at several points.

This was a useful correction to my own instinct. Once a technique works, it is tempting to promote it to recipe hygiene. The evidence instead says that entropy annealing composes with architecture-specific optimization. It does not replace it.

## Diagnostics I would record

Alongside energy and variance, an annealed run should log:

- \\(T(t)\\);
- Shannon entropy or mean \\(-\log p\\);
- maximum configuration probability;
- unique-sample count;
- effective sample support;
- sign or symmetry diagnostics;
- the epoch at which \\(T\\) reaches zero.

The post-anneal window is essential. A low regularized objective at \\(T>0\\) is not yet a low ground-state energy.

## The intuition I keep

Entropy annealing does not tell the model which quantum state is correct. It prevents an early guess from becoming irreversible.

That modest role is powerful. It changes the optimization landscape long enough for energy and phase information to remain visible. Then it gets out of the way.

## Further reading

- [Jaynes, *Information theory and statistical mechanics*](https://doi.org/10.1103/PhysRev.106.620)
- [Sharir et al., *Deep autoregressive models for variational simulation*](https://arxiv.org/abs/1902.04057)
- [Szabó and Castelnovo, *Neural network wave functions and the sign problem*](https://doi.org/10.1103/PhysRevResearch.2.033075)

