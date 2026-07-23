---
author: Faith O. Oyedemi
title: "Sites, Patches, and Hierarchies: What Does a Neural Quantum State See?"
date: 2026-01-07
draft: true
tags:
- representation learning
- patches
- neural quantum states
---

{{< katex >}}

A square lattice arrives as a physical object. A neural network receives a sequence of numbers. Between those two descriptions lies a design choice that is easy to underestimate: what counts as one token?

In my experiments I use three answers: one site, one patch, or a hierarchy of patches and sites. They do not merely change execution speed. They change the problem presented to the network.

## Site representation

For an \\(L_x\times L_y\\) lattice, site mode produces a sequence of

$$
N=L_xL_y
$$

binary tokens. Under row-major ordering, token \\(i=xL_y+y\\) represents site \\((x,y)\\).

This is the cleanest representation. Every conditional output is a Bernoulli distribution for one spin:

$$
p(\sigma_i\mid\sigma_{<i}).
$$

The output head stays small, constraints can be updated one spin at a time, and the physical meaning of a token is transparent.

The cost is sequence length. An \\(8\times8\\) lattice has 64 decisions; a \\(16\times16\\) lattice has 256. Token-mixing operations and attention may become expensive, and an RNN must carry information over long separations in the ordering.

## Patch representation

Suppose we group a \\(2\times2\\) block into one token. Each patch contains

$$
S=2\times2=4
$$

spins and therefore has \\(2^S=16\\) possible outcomes. The sequence becomes four times shorter, but each conditional distribution has 16 categories.

For a patch with \\(S\\) spins,

$$
p(\tau_k\mid\tau_{<k}), \qquad \tau_k\in\{0,\ldots,2^S-1\}.
$$

This is reminiscent of dividing an image into tokens before a Vision Transformer or MLP-Mixer. [Dosovitskiy et al.](https://arxiv.org/abs/2010.11929) and [Tolstikhin et al.](https://arxiv.org/abs/2105.01601) showed how useful patch tokenization can be in vision. A quantum wavefunction, however, must normalize and sample the categorical outcome, not merely classify the finished image.

Patching exchanges sequence length for output complexity. The number of outcomes grows exponentially:

| Patch | Spins | Outcomes |
|---|---:|---:|
| \\(1\times1\\) | 1 | 2 |
| \\(2\times2\\) | 4 | 16 |
| \\(2\times3\\) | 6 | 64 |
| \\(4\times4\\) | 16 | 65,536 |

This is why patch size is not a harmless hyperparameter. A large patch may shorten the sequence while making its head and constrained sampling impractical.

## Hierarchical representation

A hierarchy asks two models to cooperate.

The outer model chooses a coarse patch-level object. An inner subsampler resolves the spins inside that patch. Schematically,

$$
p(\sigma)= \prod_{k=1}^{K} p_{\mathrm{outer}}(z_k\mid z_{<k}) \, p_{\mathrm{inner}}(\sigma_{P_k}\mid z_{\le k},\sigma_{<P_k}).
$$

The exact factorization depends on the implementation, but the intention is stable: learn long-range structure at a coarse scale and short-range detail locally.

This is attractive for lattices with structure on more than one length scale. It also creates new failure modes. If the outer token discards information needed by the inner sampler, the hierarchy becomes a bottleneck. If only the outer model’s parameters are counted, the comparison becomes unfair because the subsampler is part of the wavefunction.

In my budgeting rules, the canonical hierarchical parameter count is:

$$
P_{\mathrm{total}}= P_{\mathrm{outer}}+P_{\mathrm{subsampler}}.
$$

## Representation changes the inductive bias

Consider a nearest-neighbour bond crossing a patch boundary. In site mode it is simply another pair of spins. In patch mode, it links two categorical tokens. Within a patch, several local correlations can be represented directly by one categorical decision.

Patches therefore favour intra-patch structure. This can help if patches align with important motifs. It can also manufacture an artificial boundary.

The same applies to lattice symmetries. A fixed \\(2\times2\\) tiling is not invariant under every one-site translation. A row-major order is not rotation invariant. Positional encodings, symmetry averaging, or multiple orderings can reduce these biases, but each changes compute or model semantics.

## Why equal parameter counts are not enough

Suppose a site model and a patch model both contain 200,000 trainable parameters. They still differ in:

- number of autoregressive decisions;
- categorical head size;
- sampling FLOPs;
- memory traffic;
- constraint handling;
- locality exposed to the architecture.

This is why I define a benchmark track using Hamiltonian, lattice, representation mode, patch shape, phase policy, and constraint policy. I compare parameter-matched models *within* that track. If I compare tracks, the representation tradeoff itself becomes the experimental variable.

## A question I use when choosing tokens

I ask:

> Which correlations should be easy before the model has learned anything?

Site tokens assume very little but create long sequences. Patches make local motifs easy and boundary-crossing structure harder. Hierarchies assume that coarse and fine information can be separated.

There is no universally correct answer. There is only an answer whose assumptions should be made visible.

## Further reading

- [Dosovitskiy et al., *An Image Is Worth 16x16 Words*](https://arxiv.org/abs/2010.11929)
- [Tolstikhin et al., *MLP-Mixer*](https://arxiv.org/abs/2105.01601)
- [Hibat-Allah et al., *Recurrent neural network wave functions*](https://doi.org/10.1103/PhysRevResearch.2.023358)

