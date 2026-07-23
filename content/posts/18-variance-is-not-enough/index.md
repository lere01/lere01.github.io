---
author: Faith O. Oyedemi
title: "Low Variance Is Not Proof of the Ground State"
date: 2026-02-25
draft: true
tags:
- energy variance
- convergence
- variational methods
---

{{< katex >}}

There is a seductive moment in variational training when the energy curve flattens and the variance falls smoothly toward zero. It looks like arrival.

But arrival where?

The energy variance answers whether a state is close to an eigenstate. It does not tell us whether that eigenstate is the ground state.

## The exact statement

For a normalized state,

$$
\sigma_H^2 = \langle H^2\rangle-\langle H\rangle^2 = \|(H-E)|\psi\rangle\|^2,
$$

where \\(E=\langle H\rangle\\).

Therefore,

$$
\sigma_H^2=0 \quad\Longleftrightarrow\quad H|\psi\rangle=E|\psi\rangle.
$$

No ground-state condition appears. Every exact excited state also has zero variance.

For a spectral expansion

$$
|\psi\rangle=\sum_n c_n|E_n\rangle,
$$

the variance is

$$
\sigma_H^2 = \sum_n|c_n|^2(E_n-E)^2.
$$

It measures spectral spread around the current mean. It does not tell us whether omitted weight lies above or below that mean.

## Why the trap matters in practice

In the \\(J_1-J_2\\) experiments, some neural states collapsed onto classical configurations or metastable basins. Their energy remained far above the reference while variance became small and stable.

One trapped trajectory even displayed a convincing power-law relation between error proxies and variance. If I had inspected variance scaling without a trustworthy energy reference or physical observables, I could have mistaken the wrong eigenstate-like basin for convergence.

The lesson is not that variance is bad. Variance is answering exactly the question defined by its equation. The mistake is asking it to identify the bottom of the spectrum.

## The V-score boundary

The [V-score](https://doi.org/10.1126/science.adg9774) uses an intensive normalization of energy variance to compare variational states across systems. It is valuable when exact energies are unavailable and for studying systematic improvement.

Its authors explicitly note the eigenstate limitation. A low V-score is evidence of low relative variance, not a certificate of ground-state identity.

I therefore think of energy and variance as two coordinates:

- energy indicates vertical position relative to a known or bounded reference;
- variance indicates spectral concentration.

When the reference is unknown, we need additional directional information.

## Diagnostics that can expose a wrong basin

On small lattices:

- exact energy error;
- fidelity with the ground state;
- spectral decomposition;
- weighted sign agreement.

On larger lattices:

- symmetry-sector weights;
- total-spin expectation;
- order parameters and structure factors;
- consistency across sizes and boundary conditions;
- independent lower or upper bounds;
- agreement across physically different initializations;
- response to perturbations or continuation.

No single observable is universal. The diagnostic should be sensitive to the failure mode one is likely to encounter.

## Can a higher moment add direction?

The third central moment is

$$
\kappa_3=\langle(H-E)^3\rangle.
$$

The ratio

$$
T=\frac{\kappa_3}{\sigma_H^2}
$$

has units of energy. For a state dominated by one eigenstate with a small contaminating component, \\(T\\) approaches the contaminant’s energy offset from the dominant state. A negative value would indicate residual spectral weight below the current dominant energy.

This connects to the Horn-Weinstein connected-moments expansion of imaginary-time energy. It is an intriguing directional diagnostic because variance discards precisely this asymmetry.

I must be careful here: this is a research proposal, not yet an established result from my campaign. A third moment of sampled local energies is also not automatically equal to the operator moment \\(\langle(H-E)^3\rangle\\). The exact operator quantity requires nested Hamiltonian action or exact enumeration.

The validation sequence should be:

1. construct healthy and trapped \\(4\times4\\) states;
2. compute exact spectral moments;
3. compare \\(T\\) with known weight above and below;
4. test whether a local-energy proxy preserves the sign;
5. only then promote it to a production diagnostic.

## A convergence checklist

Before calling a state converged, I ask:

1. Has the energy stopped changing within uncertainty?
2. Is the variance small?
3. Is the result stable across sample count and optimizer settings?
4. Do independent seeds find the same physical state?
5. Are symmetry and order diagnostics sensible?
6. Does a small-system version agree with exact calculation?
7. Is the result below known competing states and consistent with bounds?

Variance belongs on this list. It should not replace the list.

## The broader lesson

A metric can be mathematically correct and scientifically insufficient. The responsibility lies in matching the question to the claim.

Low variance tells me, “the state has stopped spreading across energies.” It does not tell me, “there is nowhere lower to go.”

That missing direction is exactly where optimization traps learn to hide.

## Further reading

- [Wu et al., *Variational benchmarks for quantum many-body problems*](https://doi.org/10.1126/science.adg9774)
- [Horn and Weinstein, *The t expansion: A nonperturbative analytic tool for Hamiltonian systems*](https://doi.org/10.1103/PhysRevD.30.1256)
- [Claudino et al., *Quantum simulations employing connected moments expansions*](https://doi.org/10.1063/5.0030688)

