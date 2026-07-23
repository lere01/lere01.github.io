---
author: Faith O. Oyedemi
title: "One Network for a Phase Diagram? Why Continuation Won"
date: 2026-02-18
draft: true
tags:
- meta-learning
- continuation
- FiLM
---

{{< katex >}}

The proposition was attractive: instead of training one neural quantum state for every \\(J_2\\), train a single wavefunction

$$
\psi_\theta(\sigma;J_2)
$$

that understands the entire coupling sweep.

If it worked, the network could interpolate between trained couplings, provide a warm start at unseen points, and perhaps reveal a shared representation of the phase diagram. I tested this with feature-wise linear modulation. The mechanism worked. The scientific strategy did not win.

## Conditioning with FiLM

[Feature-wise linear modulation](https://arxiv.org/abs/1709.07871), or FiLM, lets a conditioning variable generate scale and shift values for hidden features:

$$
h'_{\ell,c} = \gamma_{\ell,c}(J_2)h_{\ell,c} + \beta_{\ell,c}(J_2).
$$

A small generator network maps \\(J_2\\) to \\(\gamma\\) and \\(\beta\\) for each layer and channel. I initialized the modulation as an identity:

$$
\gamma=1,\qquad\beta=0.
$$

At initialization, the conditioned DPMixer therefore matched its base model exactly. Tests checked log-wavefunction equality, sampling consistency, evaluator parity, and gradient flow after modulation became active.

The model was trained round-robin over a ten-point \\(J_2\\) grid. The total 50,000-epoch budget matched the total work of one sequential continuation sweep.

## The competing strategy

Continuation is simpler:

1. train a model at \\(J_2^{(k)}\\);
2. use its converged parameters to initialize \\(J_2^{(k+1)}\\);
3. continue across neighbouring coupling values.

This assumes nearby ground states occupy nearby regions of parameter space, except perhaps near sharp changes where smaller steps or multiple paths are needed.

Continuation does not produce one universal model. It produces a family of checkpoints connected by a path.

## The held-out test

The conditioned model was evaluated at \\(J_2=0.3\\) and \\(0.7\\), neither of which appeared in training.

| Coupling | FiLM zero-shot | FiLM + 1k fine-tune | neighbour + 1k | neighbour + 5k |
|---:|---:|---:|---:|---:|
| 0.3 | 8.61% | 6.83% | 0.35% | 0.04% |
| 0.7 | 12.61% | 8.60% | 1.13% | 0.82% |

The FiLM model interpolated smoothly: held-out errors remained within the band of neighbouring trained tasks. This is evidence that conditioning mechanically worked.

The problem was the band itself. Trained tasks plateaued around 7.7-13.8%, while comparable single-task models reached roughly 0.1-3%. The shared backbone was learning a smooth family of compromises.

## Multi-task interference

One modest backbone was asked to represent ground states across three physical regimes. The modulation added only a small number of condition-dependent degrees of freedom per layer. Gradients from different \\(J_2\\) values competed for shared features.

Zero-shot interpolation therefore answered:

> Can the network interpolate between the compromises it learned?

It did not answer:

> Can the network match the best independently optimized state at each coupling?

At the tested capacity, it could not.

## Fine-tuning exposed the initialization quality

Perhaps the conditioned network was still a useful starting manifold. The matched fine-tuning test rejected that hope at this capacity.

After 1,000 epochs, a converged neighbouring checkpoint outperformed the conditioned initialization by roughly twenty-fold in relative error at both held-out couplings.

This result favours continuation for the practical task I actually face: tracing a one-dimensional coupling sweep under a fixed benchmark budget.

## What this result does not prove

The experiment used:

- one principal backbone capacity;
- one seed for the central comparison;
- a \\(4\times4\\) lattice;
- one FiLM generator design;
- a discrete sign-gauge switch supplied as input.

A much wider shared model could reduce interference. A mixture of experts could allocate capacity by phase. Condition-specific phase heads might be more effective than modulating every feature. More seeds are needed for uncertainty.

I therefore state the conclusion narrowly:

> For this one-dimensional \\(J_2\\) sweep at the tested benchmark capacity and compute, sequential warm-start continuation was decisively more effective than the FiLM-conditioned model.

## Why I value the negative result

The attractive idea was not useless. It separated smooth interpolation from physical accuracy and showed that a conditioned model can generalize continuously while remaining uniformly mediocre.

That is a warning for surrogate models of phase diagrams. Smoothness is not fidelity. A network can learn the shape of its own approximation error.

Continuation won because it lets the full parameter budget specialize at every point while borrowing a nearby solution. Sometimes the less elegant model family is the more faithful scientific instrument.

## Further reading

- [Perez et al., *FiLM: Visual Reasoning with a General Conditioning Layer*](https://arxiv.org/abs/1709.07871)
- [Bengio et al., *Curriculum Learning*](https://doi.org/10.1145/1553374.1553380)

