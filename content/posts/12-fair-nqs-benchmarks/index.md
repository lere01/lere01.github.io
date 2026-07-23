---
author: Faith O. Oyedemi
title: "What Would a Fair Neural Quantum State Benchmark Measure?"
date: 2026-01-14
draft: true
tags:
- benchmarking
- neural quantum states
- uncertainty
---

{{< katex >}}

“Model A reached a lower energy than Model B” sounds decisive. It is not, unless both models were asked the same question with comparable resources.

This concern eventually became the organizing principle of my benchmark platform. I do not want to compare architecture names. I want to compare complete experiments.

## The benchmark unit

The unit I use is

$$
\text{architecture} \times\text{Hamiltonian} \times\text{lattice} \times\text{phase point} \times\text{parameter budget} \times\text{seed}.
$$

Remove any factor and an apparent ranking can change meaning.

A model may excel at one coupling and fail near a phase transition. A larger model may win because it has ten times as many parameters. One seed may land in a fortunate optimization basin. A patch model may have fewer autoregressive steps but a much larger output head.

The benchmark should expose these conditions rather than average them into invisibility.

## Count the instantiated model

Analytic formulas are useful for proposing a hidden dimension. They are not the final parameter count.

The canonical value is

```python
sum(
    parameter.numel()
    for parameter in model.parameters()
    if parameter.requires_grad
)
```

This catches embeddings, phase heads, biases, context processors, and hierarchical subsamplers that a simplified formula may omit.

I normally target a budget and accept models within a declared tolerance, such as \\(\pm2\%\\). If a candidate misses, it should be rejected or labelled out of budget.

## Fix what defines the representation

During an ordinary budget search, I allow widths and layer counts to move. I do not silently change:

- site, patch, or hierarchical mode;
- patch dimensions;
- sign prior;
- magnetization constraint;
- phase-head policy;
- subsampler family.

These alter the physical prior or computational task. They deserve dedicated ablations.

This is especially important for patch models because a patch of \\(S\\) spins produces \\(2^S\\) outcomes. Changing patch size to hit a parameter target changes far more than capacity.

## Energy error needs uncertainty

Let \\(e\\) be the estimated energy density, \\(e_{\mathrm{ref}}\\) a reference, and \\(s_e\\) the standard error. The raw error is

$$
\delta e=e-e_{\mathrm{ref}}.
$$

A ranking based on \\(|\delta e|\\) alone can reward noise. At minimum I want:

- the estimate;
- its standard error;
- the reference and its uncertainty;
- absolute and relative error;
- the number of independent samples.

A conservative summary can use

$$
|\delta e|+1.96\,s_e,
$$

with the reference uncertainty combined when it is non-negligible. This is not a universal confidence guarantee, but it makes uncertainty visible in the score.

## Accuracy is not the whole cost

Two models can reach the same error with very different computational demands. Useful records include:

- trainable parameters;
- optimizer steps;
- number of samples;
- wall time;
- peak memory;
- estimated forward and backward FLOPs;
- energy evaluations required to reach a threshold.

The comparison I care about is computational efficiency under a parameter budget. One possible curve is error versus cumulative FLOPs. Another is the cost required to cross a fixed error threshold.

Parameter count measures capacity imperfectly. FLOPs measure arithmetic imperfectly. Wall time depends on hardware. Keeping all three is more honest than pretending one scalar contains the complete story.

## Phase grids matter

An architecture may appear excellent if evaluated only where the ground state resembles its preferred prior.

For TFIM, the grid should cover weak field, critical behaviour, and strong field. For \\(J_1-J_2\\), it should cross Néel, frustrated, and stripe regimes. For Rydberg systems, it should traverse physically meaningful detuning or blockade conditions.

I prefer reporting both per-point results and an aggregate. The per-point view shows failure structure. The aggregate answers whether performance is broad.

## Seeds are part of the physics workflow

In non-convex optimization, a seed can determine the initial parameters, sample stream, and eventual basin. Reporting only the best seed answers:

> Did this method ever work?

Reporting the distribution answers:

> How reliably does it work?

Both can be useful, but they are not interchangeable. In my sign-learning studies, basin selection is itself part of the phenomenon, so suppressing seed variation would suppress the science.

## The V-score and its boundary

[Wu et al.](https://doi.org/10.1126/science.adg9774) proposed the V-score to compare variational states using energy variance normalized into an intensive measure. It enables scaling comparisons even when exact ground-state energies are unavailable.

Variance has a precise limitation: it vanishes for every eigenstate. A metastable near-eigenstate can therefore obtain an impressive variance score while remaining far above the ground state. I still value variance, but I do not let it certify ground-state identity by itself.

## Reproducibility is a benchmark output

Every run should emit its resolved configuration, not merely the template from which it was launched. I want to know:

- the exact Hamiltonian and geometry;
- model constructor arguments;
- parameter count;
- seed;
- software version;
- device and precision;
- optimizer and schedule;
- sample count;
- checkpoint provenance;
- reference-energy source.

A benchmark is not a leaderboard produced by a script. It is a collection of scientific claims whose conditions can be reconstructed.

## The question behind the table

My goal is not to crown a permanent winning architecture. It is to identify where each combination of representation and optimization spends computation effectively, where it fails, and whether the failure persists across physics regimes.

That makes the benchmark less dramatic than a single bold number. It also makes it far more useful.

## Further reading

- [Wu et al., *Variational benchmarks for quantum many-body problems*](https://doi.org/10.1126/science.adg9774)
- [NetKet 3: Machine Learning Toolbox for Many-Body Quantum Physics](https://doi.org/10.21105/joss.03075)
- [Pineau et al., *Improving Reproducibility in Machine Learning Research*](https://arxiv.org/abs/2003.12206)

