---
author: Faith O. Oyedemi
title: "Neural Quantum States: Teaching a Network to Speak Wavefunction"
date: 2025-12-03
draft: false
tags:
- neural quantum states
- variational Monte Carlo
- quantum physics
---

{{< katex >}}

Every neural network is a function. With non-linear activation, it can learn any class of funtions . Otherwise, it is one giant linear function no matter how many layers you stack. So, what is a *neural quantum state*? It is simply a neural network intended to learn a wave function. So instead of classifying an image or predicting the next word, it learns to represent the coefficients of a quantum state.

That small change in job description opens a surprisingly large door.

## The object we are trying to represent

Consider \\(N\\) spin-\\(\tfrac12\\) particles. In the computational basis, a configuration is a string

$$
\sigma=(\sigma_1,\ldots,\sigma_N), \qquad \sigma_i\in\{-1,+1\}.
$$

A general pure state is

$$
|\psi\rangle=\sum_{\sigma}\psi(\sigma)|\sigma\rangle.
$$

The complex number \\(\psi(\sigma)\\) is the wavefunction amplitude assigned to configuration \\(\sigma\\). If we tried to store every coefficient directly, we would need \\(2^N\\) complex numbers. For 40 spins that is already more than a trillion coefficients - 1,099,511,627,776. This exponential growth is the wall that every many-body method must negotiate.

A neural quantum state replaces the table of coefficients with a parameterized function:

$$
\psi(\sigma)\longrightarrow\psi_\theta(\sigma),
$$

where \\(\theta\\) denotes the trainable network parameters. The hope is not that every possible state becomes cheap. The hope is that **physically relevant** states possess enough structure for a network to compress them.

This idea became widely known through the work of [Carleo and Troyer](https://doi.org/10.1126/science.aag2302), who represented quantum states with restricted Boltzmann machines and optimized them variationally. Neural networks have since been joined by recurrent models, Transformers, convolutional networks, and Mixers. This is in no way an exhaustive list.

## Probability is only half the wavefunction

Quantum measurement gives us the Born probability

$$
p_\theta(\sigma)=\frac{|\psi_\theta(\sigma)|^2} {\langle\psi_\theta|\psi_\theta\rangle}.
$$

It is tempting to conclude that the network only needs to learn this probability distribution. That works for amplitudes and is sufficient for stoquastic hamiltonians. But in general, a wavefunction may also carry signs or complex phases. Therefore, a useful decomposition is

$$
\begin{align}
\log\psi_\theta(\sigma) &= \sqrt{p_{\theta}(\sigma)} e^{i \phi_{\theta}(\sigma)} \\
 &= \frac{1}{2} \log p_\theta(\sigma)+i\phi_\theta(\sigma).
\end{align}
$$

The first term controls how much probability a configuration receives. The phase \\(\phi_\theta\\) controls interference. Two states can assign exactly the same probabilities to every computational-basis configuration and still have different energies because their phases differ.

This distinction becomes critical in frustrated magnets. The amplitude can look plausible while the sign structure is wrong. I will return to that problem later because it has become one of the most persistent characters in my own experiments.

## How does the network learn a ground state?

The variational principle gives us the objective:

$$
E_\theta= \frac{\langle\psi_\theta|H|\psi_\theta\rangle} {\langle\psi_\theta|\psi_\theta\rangle} \ge E_0,
$$

where \\(H\\) is the Hamiltonian and \\(E_0\\) is its ground-state energy. We adjust \\(\theta\\) to lower \\(E_\theta\\).

For large systems, evaluating the sum over all configurations is impossible. Variational Monte Carlo rewrites the energy as an expectation over \\(p_\theta\\):

$$
E_{\theta} = E_{\sigma \sim p_\theta} \left [ E_{\mathrm{loc}}(\sigma) \right ],
$$

with local energy

$$
E_{\mathrm{loc}}(\sigma) = \sum_{\sigma '} H_{\sigma\sigma '} \frac{\psi_\theta(\sigma ')}{\psi_\theta(\sigma)}.
$$

We draw configurations, evaluate their local energies, estimate a gradient, and update the network. The neural network is therefore both a compressed wavefunction and a trainable sampling model.

## Autoregressive models

An autoregressive state writes the probability as

$$
p_\theta(\sigma)= \prod_{i=1}^{N} p_\theta(\sigma_i\mid\sigma_1,\ldots,\sigma_{i-1}).
$$

Each conditional distribution is normalized, so the entire probability is normalized. Sampling becomes a forward construction: choose the first spin, use it to choose the second, and continue until the lattice is complete.

This avoids the warm-up, autocorrelation, and mixing questions of a local Markov chain. It does not make sampling free, and it does not guarantee a good wavefunction. It does give us an exact sample from the distribution represented by the model. [Sharir and collaborators](https://arxiv.org/abs/1902.04057) showed why this property is so attractive for variational simulations.

In my work, the autoregressive family includes RNNs, Transformers, and Mixers, each operating at the level of sites, patches, or a hierarchy of both. The architecture changes how information moves. The physical contract does not: the model must return a consistent amplitude, phase, and sampler for the same state.

## What a neural quantum state does not promise

There are three promises I do not want to smuggle into the name.

First, expressiveness is not trainability. A network may be capable of representing a state that optimization never finds.

Second, a low energy variance only tells us that the state is close to *an* eigenstate. It does not prove that it is the ground state.

Third, a larger network is not automatically a fairer competitor. Representation, sampling cost, parameter count, and computational budget must all be controlled.

These caveats are not peripheral. They are where much of the interesting research begins.

## The picture I now carry

I think of a neural quantum state as three objects held together:

1. a representation of \\(\log\psi_\theta(\sigma)\\);
2. a probability model from which configurations can be sampled;
3. a variational object whose parameters can be moved through Hilbert space.

If any of the three disagrees with the others, the calculation is scientifically unsound. If they agree, we gain a flexible laboratory for asking how architecture, physical priors, and optimization interact.

That is the laboratory behind the articles that follow.

## Further reading

- [Carleo and Troyer, *Solving the quantum many-body problem with artificial neural networks*](https://doi.org/10.1126/science.aag2302)
- [Carleo et al., *Machine learning and the physical sciences*](https://doi.org/10.1103/RevModPhys.91.045002)
- [Sharir et al., *Deep autoregressive models for the efficient variational simulation of many-body quantum systems*](https://arxiv.org/abs/1902.04057)

