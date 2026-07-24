---
author: Faith O. Oyedemi
title: "The Multilayer Perceptron as a Potential Neural Quantum State (Part Two)"
date: 2025-12-18
draft: false
tags:
- MLP-Mixer
- neural quantum states
- autoregressive models
- quantum physics
---

{{< katex >}}

In [Part One](/posts/05-mixer_architecture/), we saw the MLP-Mixer in its original setting: image classification. The model divided an image into patches, mixed information across patches and channels, pooled the resulting representation, and predicted a class.

A neural quantum state needs a different contract. Given a spin configuration \\(\sigma\\), it must assign a complex amplitude \\(\psi_\theta(\sigma)\\). If it is autoregressive, it must also generate configurations from the same probability distribution used to evaluate that amplitude.

The Mixer can do this, but not by changing only its final layer.

## What has to change?

The image-classification Mixer answers one global question: which label belongs to this image? A spin wavefunction answers a sequence of conditional questions:

$$ p_\theta(\sigma)=\prod_{i=1}^{N}p_\theta(\sigma_i\mid \sigma_{<i}). $$

For a two-level site, the model produces two logits at step \\(i\\). A softmax converts them into the conditional probabilities for \\(\sigma_i=0\\) and \\(\sigma_i=1\\). The amplitude can then be written as

$$ \log\psi_\theta(\sigma)=\frac12\sum_{i=1}^{N}\log p_\theta(\sigma_i\mid\sigma_{<i})+i\phi_\theta(\sigma). $$

This requires four structural changes to the original Mixer:

1. replace image patches with physical site or patch states e.g. replace image tensors with spin tensors;
2. impose a generation order and prevent information from leaking backwards through it i.e. enforce causality;
3. replace global pooling and classification with conditional amplitude outputs (hint: involves a softmax operation);
4. provide both `logpsi` and `sample` operations for the same learned state.

## From image patches to physical tokens

For a spin-\\(\tfrac12\\) lattice, the smallest token is one site:

$$ \sigma_i\in\{0,1\}. $$

Each value is mapped to a learned embedding and combined with a representation of its position. A special beginning-of-sequence token supplies the context for the first decision.

Larger tokens are also possible. A patch containing \\(S\\) binary sites has \\(2^S\\) possible states and can be treated as one categorical variable. Patching shortens the autoregressive sequence, but the output head grows exponentially with \\(S\\). That trade-off is a modelling decision, not a free speed-up.

The Mixer’s two axes now have a physical interpretation:

- token mixing moves information between earlier sites or patches;
- channel mixing transforms the learned features within each token.

For the eight-atom chain in Part Three, we will use one site per token. This keeps the mapping between atoms and conditional decisions transparent.

## Causality is the essential modification

The original MLP-Mixer is not autoregressive. Its token-mixing MLP can use every patch to update every other patch. That is appropriate when the complete image is already available, but invalid when predicting the next spin.

At step \\(i\\), the network may depend on \\(\sigma_1,\ldots,\sigma_{i-1}\\), but **not** on \\(\sigma_i\\) or any later value. In schematic form, the allowed dependency pattern is lower triangular:

```text
prediction       visible spin values
sigma_1          -
sigma_2          sigma_1
sigma_3          sigma_1 sigma_2
sigma_4          sigma_1 sigma_2 sigma_3
```

So, we make the token mixing causal and evaluate the state in a fixed lattice order. For a one-dimensional open chain, left to right is the natural choice. On a two-dimensional lattice, row-major and snake orderings are both possible, but they give the model different inductive biases.

A useful implementation test is the prefix test: changing a future spin must not change an earlier conditional distribution. Passing this test matters more than whether a tensor has the expected shape. A network can run successfully and still leak future information.

## The amplitude head

The classifier in Part One pooled all tokens and returned class logits. The autoregressive model instead returns conditional logits at every generation step:

```python
# Schematic only: this is the public interface, not the research implementation.
prefix = begin_sequence()
log_probability = 0.0

for site in ordering:
    hidden = causal_mixer(prefix, site)
    probabilities = softmax(amplitude_head(hidden))
    log_probability += log(probabilities[configuration[site]])
    prefix = append(prefix, configuration[site])

log_amplitude = 0.5 * log_probability
```

The factor of one half appears because Born probabilities are squared amplitudes:

$$ |\psi_\theta(\sigma)|^2=p_\theta(\sigma). $$

Since every conditional distribution is normalized, their product is normalized too. This is one of the main attractions of an autoregressive neural quantum state: it provides direct samples without a Markov-chain accept-reject process.

## What about signs and phases?

Probabilities alone do not specify a general wavefunction. So we can allow a second output to represent phase:

$$ \psi_\theta(\sigma)=\sqrt{p_\theta(\sigma)}\exp^{i\phi_\theta(\sigma)}. $$

For some stoquastic Hamiltonians, the ground state can be chosen real and non-negative in the computational basis. In that setting, an amplitude-only demonstration is appropriate. Frustrated or genuinely complex states require more care: a model can reproduce the probability distribution while learning the wrong interference pattern.

The eight-atom Rydberg example in Part Three is deliberately amplitude-only. That is a property of the chosen demonstration, not a claim that phase modelling is optional in general.

## Evaluation and sampling must agree

An autoregressive wavefunction needs two complementary operations:

- `logpsi(configuration)` accumulates the selected conditional log-probabilities and phase;
- `sample(batch_size)` draws each site from those same conditionals.

They must describe exactly the same distribution. This sounds obvious, but differences in ordering, constraints, token encoding, or masking can make evaluation and sampling silently disagree.

For a small system, I can check normalization directly:

$$ \sum_{\sigma\in\{0,1\}^N}\exp\!\left(2\,\mathrm{Re}\log\psi_\theta(\sigma)\right)=1. $$

For \\(N=8\\), that sum contains only 256 configurations. I can also compare empirical sample frequencies with the probabilities returned by `logpsi`. These tests become impractical at scale, which makes the small-system check especially valuable.

## Connecting the state to physics

The Mixer is only an ansatz until it is paired with a Hamiltonian. The variational objective is

$$ E_\theta=\frac{\langle\psi_\theta|H|\psi_\theta\rangle}{\langle\psi_\theta|\psi_\theta\rangle}\ge E_0. $$

In a large calculation, variational Monte Carlo estimates this energy from configurations drawn from \\(p_\theta\\). The local energy is

$$ E_{\mathrm{loc}}(\sigma)=\sum_{\sigma'}H_{\sigma\sigma'}\frac{\psi_\theta(\sigma')}{\psi_\theta(\sigma)}. $$

This is where architecture, sampling, and the physical model finally meet. The Mixer proposes conditional probabilities; the Hamiltonian evaluates the state those probabilities define; optimization changes the network parameters to lower the energy.

## The implementation boundary

The code above is intentionally schematic. It exposes the architectural contract without reproducing the current research implementation, optimization schedule, caching strategy, or benchmarking apparatus. Those details belong with the completed study, where they can be documented together and interpreted in context.

The public claims here are narrower:

- a Mixer can be made autoregressive by enforcing causal token dependencies;
- normalized conditional outputs define the wavefunction amplitude;
- a phase output can extend the ansatz beyond non-negative states;
- consistent evaluation and direct sampling make the model usable in variational Monte Carlo.

[Part Three](/posts/22-mixer-rydberg-chain/) puts that contract through a small test: an open chain of eight interacting Rydberg atoms, where the complete Hilbert space is still available as a reference.

## Further reading

- [Tolstikhin et al., *MLP-Mixer: An all-MLP Architecture for Vision*](https://proceedings.neurips.cc/paper_files/paper/2021/file/cba0a4ee5ccd02fda0fe3f9a3e7b89fe-Paper.pdf)
- [Sharir et al., *Deep Autoregressive Models for the Efficient Variational Simulation of Many-Body Quantum Systems*](https://arxiv.org/abs/1902.04057)
- [Carleo and Troyer, *Solving the Quantum Many-Body Problem with Artificial Neural Networks*](https://doi.org/10.1126/science.aag2302)
