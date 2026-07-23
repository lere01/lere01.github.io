---
author: Faith O. Oyedemi
title: "From Ground States to Motion: TDVP for Neural Wavefunctions"
date: 2026-03-04
draft: true
tags:
- TDVP
- quantum dynamics
- neural quantum states
---

{{< katex >}}

Ground-state optimization asks us to descend an energy landscape. Quantum dynamics asks the wavefunction to move without descending at all:

$$
i\frac{d}{dt}|\psi(t)\rangle=H|\psi(t)\rangle.
$$

If \\(|\psi_\theta\rangle\\) is restricted to a neural-network manifold, exact evolution generally points outside that manifold. The time-dependent variational principle finds the best tangent direction available inside it.

## Projecting Schrödinger’s equation

For parameters \\(\theta(t)\\),

$$
\frac{d}{dt}|\psi_\theta\rangle = \sum_k\dot\theta_k \frac{\partial|\psi_\theta\rangle}{\partial\theta_k}.
$$

McLachlan’s principle minimizes the residual

$$
\left\| \frac{d}{dt}|\psi_\theta\rangle +iH|\psi_\theta\rangle \right\|^2
$$

over \\(\dot\theta\\). For real parameters and complex log derivatives, this produces a linear system built from the same quantum geometric tensor encountered in stochastic reconfiguration:

$$
S\,\dot\theta=f.
$$

Real-time and imaginary-time evolution share the geometry but use different force projections. A sign error in that projection does not make the run slightly inaccurate. It evolves the state according to a different equation.

## A basis example

For TFIM in the \\(z\\) basis,

$$
H_z=-J\sum_{\langle ij\rangle}Z_iZ_j-h\sum_iX_i.
$$

After a basis rotation exchanging \\(X\\) and \\(Z\\),

$$
H_x=-J\sum_{\langle ij\rangle}X_iX_j-h\sum_iZ_i.
$$

The physical spin direction and simulation basis must be recorded separately. A \\(z\\)-polarized physical initial state may be broad in the \\(x\\) simulation basis, while an \\(x\\)-polarized state may be deterministic.

This matters numerically. A deterministic sample distribution can collapse the initial quantum geometric tensor and make the first TDVP solve singular.

## Sampling once, replaying carefully

A production time step may require local energies, log-derivative Jacobians, QGT products, and observables. Holding the full model graph for a million samples is not realistic.

The implementation I use streams exact autoregressive samples into a compact CPU replay store. The same fixed sample set is replayed in bounded chunks for:

- forces;
- dense or matrix-free QGT operations;
- observables;
- residual diagnostics.

Reusing the store prevents each quantity from seeing an unrelated Monte Carlo realization while controlling memory.

For distributed runs, samples are divided exactly across ranks and sufficient statistics are reduced explicitly. Parameters are replicated, but ordinary DDP gradient synchronization is not the governing abstraction because TDVP solves a global stochastic linear system.

## Dense and matrix-free geometry

For modest parameter counts, we can accumulate a dense parameter-space QGT. Larger models use products \\(Sv\\) and iterative solvers.

The current large-model path builds a reorthogonalized Lanczos basis, estimates QGT Ritz modes, and applies a soft signal-to-noise filter. This is an approximation, so the run records:

- filtered-system residual;
- Ritz residual;
- basis orthogonality;
- terminal recurrence coefficient;
- truncation estimate.

A time step must not advance merely because the Krylov routine terminated. It advances when the declared solve tolerance is satisfied.

## Integration is another source of error

Once \\(\dot\theta\\) is known, parameters still need a time integrator. An adaptive Heun scheme can compare one full step with two half steps.

In the implementation, an attempted step is transactional:

- a rejected step changes no physical time;
- model parameters are restored;
- RNG state is restored;
- no trajectory point is committed;
- integrated residual is unchanged.

This sounds operational, but it protects the mathematical meaning of an adaptive stochastic trajectory. If rejection consumes a new sample stream or partially updates an optimizer state, retrying at a smaller \\(dt\\) is not the same experiment.

## How a quench should be validated

At least four convergence ladders are distinct:

1. **Sampling:** increase the number of configurations.
2. **Integration:** reduce \\(dt\\).
3. **Linear solve:** tighten tolerance and increase Krylov dimension.
4. **Variational capacity:** enlarge or change the ansatz.

Seeds, disorder realizations, and lattice size add further axes.

On small systems, the full trajectory should be compared with exact evolution:

$$
|\psi(t)\rangle=e^{-iHt}|\psi(0)\rangle.
$$

On large systems, observable agreement over an overlapping time window is more informative than matching only the final point.

## Replication and adaptation

Some current configurations follow physical protocols from published quench studies while replacing their complex CNN with the native autoregressive RNN or DPMixer.

I label these results *native autoregressive adaptations*, not replications. The Hamiltonian, initial state, sample budget, and observables may match, but architecture and sampling semantics affect the approximation.

This wording is not modesty for its own sake. It tells the reader which scientific object was reproduced and which was redesigned.

## What TDVP adds to the research

Ground-state training tells us whether a network can represent and find a stationary state. Dynamics asks whether its tangent space remains faithful along an entire trajectory.

That is a harsher test. Errors accumulate, geometry changes, and every approximation has a clock.

The reward is equally large: we can ask how correlations spread, how order melts after a quench, and where a variational family ceases to track unitary motion.

## Further reading

- [Carleo et al., *Light-cone effect and supersonic correlations in one- and two-dimensional bosonic superfluids*](https://doi.org/10.1038/srep01115)
- [Schmitt and Heyl, *Quantum many-body dynamics in two dimensions with artificial neural networks*](https://doi.org/10.1103/PhysRevLett.125.100503)
- [López Gutiérrez and Mendl, *Real time evolution with neural-network quantum states*](https://doi.org/10.1103/PhysRevLett.128.020501)
- [Donatella et al., *Dynamics with autoregressive neural quantum states*](https://doi.org/10.1103/PhysRevA.108.022210)

