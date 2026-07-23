---
author: Faith O. Oyedemi
title: "Stochastic Reconfiguration: Following the Geometry of a Wavefunction"
date: 2026-01-21
draft: true
tags:
- stochastic reconfiguration
- quantum geometric tensor
- optimization
---

{{< katex >}}

Ordinary gradient descent asks which parameter direction lowers the energy fastest. Stochastic reconfiguration asks a more physical question:

> Which parameter change moves the quantum state in the right direction without taking an unnecessarily large step through Hilbert space?

The difference matters because neural-network parameters are coordinates, not physical distances.

## Unequal parameter steps

Suppose two parameter updates have the same Euclidean length,

$$
\|\delta\theta^{(1)}\|_2 = \|\delta\theta^{(2)}\|_2.
$$

They need not change the normalized wavefunction by the same amount. One parameter may barely affect \\(\psi\\); another may alter dominant configurations dramatically. Rescaling a hidden layer can also change parameter geometry without changing the represented state in an equally dramatic way.

The wavefunction itself supplies a better metric.

## Logarithmic derivatives

Define

$$
O_k(\sigma) = \frac{\partial\log\psi_\theta(\sigma)} {\partial\theta_k}.
$$

For real parameters and a complex wavefunction, a common real quantum geometric tensor is

$$
S_{kl} = \operatorname{Re} \left[ \langle O_k^*O_l\rangle -\langle O_k^*\rangle\langle O_l\rangle \right].
$$

Expectations are taken over \\(|\psi_\theta|^2\\). This is a covariance matrix of tangent vectors. It measures how distinguishable nearby normalized states are under parameter changes.

The energy force can be written

$$
F_k = 2\operatorname{Re} \left[ \langle O_k^*E_{\mathrm{loc}}\rangle -\langle O_k^*\rangle \langle E_{\mathrm{loc}}\rangle \right],
$$

up to a sign convention absorbed into the update. Stochastic reconfiguration solves

$$
(S+\lambda I)\,\delta\theta=-\eta F.
$$

Here \\(\eta\\) is a step size and \\(\lambda\\) regularizes noisy or nearly singular directions.

## Natural gradient in quantum clothing

The method is closely related to natural gradient. [Amari](https://doi.org/10.1162/089976698300017746) argued that optimization should respect the information geometry of probability distributions. [Sorella](https://doi.org/10.1103/PhysRevLett.80.4558) introduced stochastic reconfiguration in quantum Monte Carlo. In neural quantum states, the two viewpoints meet.

Another interpretation is imaginary-time evolution projected onto the variational manifold. Exact imaginary time suppresses excited-state components:

$$
|\psi(\tau)\rangle \propto e^{-\tau H}|\psi(0)\rangle.
$$

SR chooses the tangent-space motion that best approximates this descent for the current ansatz.

## Why the solve becomes difficult

If the model has \\(P\\) parameters, a dense \\(S\\) contains \\(P^2\\) entries. At \\(P=200{,}000\\), storing it is impossible.

Even for modest \\(P\\), the sample covariance may be rank-deficient when the number of samples is smaller than the number of parameters. Neural networks also contain redundant or weak directions. Regularization is therefore not optional decoration.

There are several strategies:

- dense parameter-space solves for small models;
- sample-space or minimum-SR formulations;
- matrix-free products \\(Sv\\);
- conjugate-gradient solves;
- low-rank Krylov or Lanczos approximations;
- diagonal or structured approximations.

Each changes numerical behaviour. A paper should report the solver, tolerance, regularization, and residual, not merely say “we used SR.”

## Sampling noise enters the geometry

Both \\(S\\) and \\(F\\) are estimated from samples. Small eigenvalues of \\(S\\) are particularly sensitive to noise. Inverting them can amplify a harmless fluctuation into a huge parameter update.

Adding \\(\lambda I\\) damps those directions:

$$
\frac{1}{s_i} \longrightarrow \frac{1}{s_i+\lambda}
$$

in the eigenbasis of \\(S\\). Too little regularization is unstable. Too much reduces SR toward a small Euclidean step and may erase useful curvature.

This is why I monitor the linear-system residual and update norm. A solver that returns a vector is not necessarily a solver that resolved the intended geometry.

## Amplitude and phase may move on different clocks

For a complex wavefunction, the parameter vector may contain an amplitude pathway and a phase pathway. Equal Euclidean learning rates do not guarantee equal progress in the represented state.

This matters in frustrated systems, where the model can lower the immediate objective by suppressing a badly signed configuration before correcting its phase. One research direction is to scale phase and amplitude blocks differently. Such a modification should be tested as an explicit algorithm, not hidden inside tuning.

## A minimal mental model

I think of ordinary gradient descent as following an arrow drawn on the parameter map. SR first asks how the map is stretched, then corrects the arrow so that the step has meaning in wavefunction space.

That correction can be powerful. It can also become noisy, singular, and expensive. Geometry does not remove numerical analysis; it makes the numerical assumptions easier to see.

## Further reading

- [Sorella, *Green Function Monte Carlo with Stochastic Reconfiguration*](https://doi.org/10.1103/PhysRevLett.80.4558)
- [Sorella, *Wave function optimization in the variational Monte Carlo method*](https://doi.org/10.1103/PhysRevB.71.241103)
- [Stokes et al., *Quantum Natural Gradient*](https://doi.org/10.22331/q-2020-05-25-269)
- [Carleo et al., *Machine learning and the physical sciences*](https://doi.org/10.1103/RevModPhys.91.045002)

