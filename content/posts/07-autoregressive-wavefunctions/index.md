---
author: Faith O. Oyedemi
title: "Autoregressive Wavefunctions: Building a Quantum State One Spin at a Time"
date: 2025-12-10
draft: true
tags:
- autoregressive models
- neural quantum states
- sampling
---

{{< katex >}}

Suppose I hand you a completed \\(8\times8\\) spin configuration and ask for its probability. That is one problem. Suppose instead I ask you to generate a new configuration distributed exactly according to a neural network. That is a different problem, and it is the second one that makes autoregressive wavefunctions special.

The idea comes from sequence modelling: turn one complicated joint distribution into a chain of simpler conditional decisions.

## The chain rule does the heavy lifting

For spins \\(\sigma_1,\ldots,\sigma_N\\), probability theory gives

$$
p(\sigma_1,\ldots,\sigma_N) = \prod_{i=1}^{N} p(\sigma_i\mid\sigma_{<i}),
$$

where \\(\sigma_{<i}\\) means all spins preceding site \\(i\\) in a chosen ordering.

This identity is exact. The modelling decision is to let a neural network predict each conditional probability. For a spin-\\(\tfrac12\\) system, the network produces two non-negative numbers that sum to one.

Sampling then becomes almost embarrassingly direct:

```python
sample = []
for site in ordering:
    probabilities = model.conditional(sample, site)
    sample.append(draw(probabilities))
```

There is no accept-reject step. A complete pass produces one exact sample from the distribution encoded by the model. Here, *exact* refers to the sampling procedure, not to the physical accuracy of the learned state.

That distinction is worth keeping. A model can sample its own wrong answer perfectly.

## From probability to wavefunction

For a normalized autoregressive state,

$$
|\psi_\theta(\sigma)|= \sqrt{p_\theta(\sigma)}.
$$

The phase is represented separately:

$$
\psi_\theta(\sigma) = \sqrt{p_\theta(\sigma)} e^{i\phi_\theta(\sigma)}.
$$

Some models produce a phase contribution at each step; others use a separate global head after processing the full configuration. Either design is acceptable if `sample` and `logpsi` describe the same amplitude.

This consistency sounds obvious until symmetry projection or constraints enter. If I evaluate a projected amplitude but sample from the unprojected state, the configurations are not distributed according to the probability used in my estimators. I then need a correct projected sampler or explicit importance weights. Otherwise, a beautifully implemented calculation answers the wrong expectation value.

## A lattice is not naturally a sentence

A language model inherits an ordering from text. A square lattice does not.

We may traverse it row by row:

$$
(0,0),(0,1),\ldots,(0,L-1),(1,0),\ldots
$$

or use a snake, spiral, space-filling curve, or symmetry-related ensemble of orderings. Each choice changes which dependencies appear short-range to the network.

The final joint distribution can represent correlations in any direction, but the inductive bias is not neutral. A row-major RNN sees horizontal neighbours consecutively while many vertical neighbours are separated by an entire row. A Transformer can attend across that distance, but its causal mask still respects the chosen direction.

In my code, site numbering is row-major unless a configuration says otherwise. This convention also determines how lattice sites map into basis-state bits. It is a mundane detail with physical consequences, which is why I prefer to record it rather than trust memory.

## Constraints require conditional reasoning

Consider a calculation restricted to zero magnetization. If \\(N\\) is even, every valid configuration must contain exactly \\(N/2\\) up spins.

A naive model can generate invalid configurations and reject them, but the rejection rate may become terrible. A constrained autoregressive sampler instead masks choices that would make the target impossible.

If five sites remain and four up spins are still required, the next decision is already constrained. Near the end of the sequence, it may be forced.

The resulting probability remains normalized over the allowed sector, provided masking is applied consistently during both sampling and wavefunction evaluation.

There is also a physical restriction: the chosen Hamiltonian must conserve the quantity being fixed. The Heisenberg \\(J_1-J_2\\) model conserves total \\(S^z\\). The transverse-field Ising model generally does not. Imposing a fixed number of up spins on the latter would remove states that the Hamiltonian dynamically connects.

## Exact samples are not independent computations

Autoregressive sampling removes Markov-chain autocorrelation, but samples still share a trained model and finite numerical precision. Statistical uncertainty remains. Model bias remains. Optimization bias remains.

It is also possible for the probability distribution to collapse onto a small set of configurations. Exact sampling will faithfully expose that collapse, repeatedly returning the same classical patterns. This is not a failure of the sampler. It is evidence about what the wavefunction has learned.

## Why this is computationally interesting

The cost of one sample depends on how much previous computation can be cached. An RNN carries a hidden state forward. A Transformer may cache keys and values. A Mixer designed for autoregressive use must respect causality while moving information across tokens.

Generating many samples also invites parallelism. We can advance a batch together, but production runs should still stream samples in bounded chunks. A million \\(10\times10\\) configurations stored as one byte per spin already occupy roughly 100 MB before observables or model intermediates are considered.

The right question is therefore not merely “Can the model sample exactly?” It is:

> Can the model sample the same wavefunction that it evaluates, under the correct physical constraints, at a cost that remains manageable as the lattice grows?

That is the contract I use when comparing autoregressive architectures.

## Further reading

- [Sharir et al., *Deep autoregressive models for the efficient variational simulation of many-body quantum systems*](https://arxiv.org/abs/1902.04057)
- [Hibat-Allah et al., *Recurrent neural network wave functions*](https://doi.org/10.1103/PhysRevResearch.2.023358)
- [Wang and Davis, *Calculating Rényi entropies with neural autoregressive quantum states*](https://arxiv.org/abs/2003.01358)

