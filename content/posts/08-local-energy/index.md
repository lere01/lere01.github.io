---
author: Faith O. Oyedemi
title: "Local Energy: The Quantity That Makes Variational Monte Carlo Practical"
date: 2025-12-17
draft: false
tags:
- variational Monte Carlo
- local energy
- Hamiltonians
---

{{< katex >}}

The phrase *local energy* does not mean the energy near one lattice site. It is local in configuration space: it tells us what energy the Hamiltonian assigns when we anchor the calculation at one sampled configuration.

This quantity is the hinge that turns an exponentially large quantum expectation into a Monte Carlo estimate.

## Starting from the variational energy

For a state \\(|\psi_\theta\rangle\\),

$$
E_\theta= \frac{\langle\psi_\theta|H|\psi_\theta\rangle} {\langle\psi_\theta|\psi_\theta\rangle}.
$$

Insert computational-basis states twice:

$$
E_\theta= \frac{\sum_{\sigma,\sigma'} \psi_\theta^*(\sigma) H_{\sigma\sigma'} \psi_\theta(\sigma')} {\sum_\sigma|\psi_\theta(\sigma)|^2}.
$$

Now multiply and divide each term by \\(\psi_\theta(\sigma)\\). Assuming sampled configurations have nonzero amplitude,

$$
E_\theta= \sum_\sigma p_\theta(\sigma) \sum_{\sigma'} H_{\sigma\sigma'} \frac{\psi_\theta(\sigma')}{\psi_\theta(\sigma)}.
$$

The inner sum is the local energy:

$$
E_{\mathrm{loc}}(\sigma)= \sum_{\sigma'} H_{\sigma\sigma'} \frac{\psi_\theta(\sigma')}{\psi_\theta(\sigma)}.
$$

Therefore,

$$
E_\theta=\mathbb{E}_{\sigma\sim p_\theta} [E_{\mathrm{loc}}(\sigma)].
$$

That is the trick. We do not enumerate the Hilbert space. We sample \\(\sigma\\), generate only the configurations \\(\sigma'\\) connected to it by \\(H\\), and average.

## Diagonal and off-diagonal terms

Take the transverse-field Ising model in the \\(z\\) basis:

$$
H=-J\sum_{\langle ij\rangle}Z_iZ_j-h\sum_iX_i.
$$

For a spin configuration \\(\sigma_i=\pm1\\), the Ising term is diagonal:

$$
E_{\mathrm{diag}}(\sigma) =-J\sum_{\langle ij\rangle}\sigma_i\sigma_j.
$$

The transverse field flips one spin. If \\(\sigma^{(i)}\\) is \\(\sigma\\) with site \\(i\\) flipped, then

$$
E_{\mathrm{loc}}(\sigma)= -J\sum_{\langle ij\rangle}\sigma_i\sigma_j -h\sum_i \frac{\psi_\theta(\sigma^{(i)})} {\psi_\theta(\sigma)}.
$$

This equation makes the neural network’s role concrete. It must compare the sampled configuration with every connected configuration. If the network outputs log amplitudes, the ratio can be evaluated stably as

$$
\frac{\psi(\sigma')}{\psi(\sigma)} =\exp\left[\log\psi(\sigma')-\log\psi(\sigma)\right].
$$

For the \\(J_1-J_2\\) Heisenberg model, off-diagonal terms exchange anti-aligned spins. For the Rydberg model, the laser term flips occupation while detuning and interactions contribute diagonally. The bookkeeping changes, but the local-energy pattern remains.

## Why conventions matter

Suppose two codes both claim to implement

$$
J\,\mathbf{S}_i\cdot\mathbf{S}_j.
$$

One may use Pauli matrices directly, while another uses
\\(\mathbf{S}=\boldsymbol{\sigma}/2\\). The corresponding Pauli expression is

$$
\mathbf{S}_i\cdot\mathbf{S}_j = \frac14(X_iX_j+Y_iY_j+Z_iZ_j).
$$

Missing the factor of \\(1/4\\) changes every reference energy. Counting a periodic bond twice changes the energy again. Reversing a detuning sign can exchange which occupation is favoured.

For this reason, a local-energy implementation should state:

- the simulation basis;
- site and bit ordering;
- boundary conditions;
- whether interaction pairs are unique;
- the value attached to each pair;
- the normalization of reported energy.

These are scientific definitions, not low-level software trivia.

## The zero-variance property

If \\(\psi\\) is an exact eigenstate,

$$
H|\psi\rangle=E|\psi\rangle,
$$

then \\(E_{\mathrm{loc}}(\sigma)=E\\) wherever \\(\psi(\sigma)\neq0\\). Every sample returns the same value and the variance vanishes.

This gives us a powerful diagnostic. It also contains a trap: every eigenstate has zero variance, including excited states. Low variance means the state is becoming eigenstate-like. It does not identify which eigenstate.

I used to think of energy and variance as a complete pair: energy tells us where we are, variance tells us whether we have settled. My experiments with frustrated magnets forced a refinement. A state can settle very convincingly in the wrong place.

## Estimation and uncertainty

Given \\(M\\) independent samples,

$$
\widehat E=\frac1M\sum_{m=1}^{M} E_{\mathrm{loc}}(\sigma^{(m)}).
$$

For complex wavefunctions and Hermitian \\(H\\), the true expectation is real, although finite sampling and numerical error may leave a small imaginary residual. That residual should be monitored rather than silently discarded.

The standard error scales approximately as

$$
\operatorname{SE}(\widehat E) \approx \sqrt{\frac{\operatorname{Var}(E_{\mathrm{loc}})}{M}}
$$

for independent samples. Markov-chain samples require an autocorrelation correction. Exact autoregressive samples remove that correction, but not the need to report uncertainty.

## A compact implementation pattern

```python
samples = wavefunction.sample(num_samples)
connected, matrix_elements = hamiltonian.connected(samples)

logpsi = wavefunction.logpsi(samples)
logpsi_connected = wavefunction.logpsi(connected)

ratios = (logpsi_connected - logpsi).exp()
local_energy = (matrix_elements * ratios).sum(dim=-1)

energy = local_energy.mean()
stderr = local_energy.real.std() / num_samples**0.5
```

Real code must handle batching, reshaping, constraints, complex types, and memory limits. The scientific structure, however, fits into these few lines.

The local energy is where the Hamiltonian, wavefunction, and sampler finally meet. If I want to trust a variational result, this is one of the first junctions I test against exact enumeration.

## Further reading

- [McMillan, *Ground state of liquid He-4*](https://doi.org/10.1103/PhysRev.138.A442)
- [Carleo and Troyer, *Solving the quantum many-body problem with artificial neural networks*](https://doi.org/10.1126/science.aag2302)
- [Becca and Sorella, *Quantum Monte Carlo Approaches for Correlated Systems*](https://doi.org/10.1017/CBO9781316417041)

