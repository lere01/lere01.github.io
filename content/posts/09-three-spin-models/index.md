---
author: Faith O. Oyedemi
title: "Three Spin Models, Three Different Questions"
date: 2025-12-24
draft: true
tags:
- transverse-field Ising model
- Rydberg atoms
- J1-J2 model
---

{{< katex >}}

TFIM, Rydberg, and \\(J_1-J_2\\) appear together throughout my benchmark configurations. It would be easy to treat them as three datasets presented to the same neural network. Physically, they ask rather different questions, and those differences determine which errors a neural quantum state is likely to reveal.

I find it helpful to meet the Hamiltonians before meeting the architectures.

## Transverse-field Ising model

The square-lattice transverse-field Ising model is

$$
H_{\mathrm{TFIM}} = -J\sum_{\langle ij\rangle}Z_iZ_j -h\sum_iX_i.
$$

Here \\(J>0\\) favours aligned spins along \\(z\\), while \\(h\\) tries to polarize them along \\(x\\). The two terms do not commute.

At \\(h=0\\), the ground states are simple ferromagnetic product states in the \\(z\\) basis. At very large \\(h/J\\), the ground state approaches an \\(x\\)-polarized product state. Between those limits lies a quantum phase transition.

This model is useful because the source of quantum fluctuation is explicit: the field flips one spin at a time. In the \\(z\\) basis, its off-diagonal matrix elements are non-positive for \\(h>0\\), so the ground state can be chosen non-negative. The network’s main challenge is learning amplitudes and correlations rather than a difficult sign structure.

It is therefore an excellent control problem. If an optimization method cannot solve modest TFIM lattices, frustration is probably not the first suspect.

## Rydberg atoms

A common effective Hamiltonian for a driven array is

$$
H_{\mathrm{Ry}} = -\frac{\Omega}{2}\sum_iX_i -\Delta\sum_i n_i +\sum_{i<j}V_{ij}n_in_j,
$$

where

$$
n_i=\frac{1+Z_i}{2}
$$

under the convention that bit one denotes Rydberg occupation. \\(\Omega\\) is the Rabi frequency, \\(\Delta\\) the detuning, and \\(V_{ij}\\) the interaction between excited atoms.

For van der Waals interactions,

$$
V_{ij}\propto\frac{1}{r_{ij}^6}.
$$

The competition differs from TFIM. Detuning rewards or penalizes occupation, while the interaction prevents nearby simultaneous excitations. This produces the Rydberg blockade and ordered excitation patterns.

The geometry matters directly because distance enters the coupling. Open and periodic boundaries are not cosmetic choices. Neither is truncating the interaction range. A benchmark must record the coordinates and the interaction pairs from which each \\(V_{ij}\\) was calculated.

Experiments such as the programmable 51-atom simulator reported by [Bernien and collaborators](https://doi.org/10.1038/nature24622) make this model more than a theoretical exercise. It connects variational algorithms to controllable quantum matter.

## The frustrated \\(J_1-J_2\\) Heisenberg model

The square-lattice model is

$$
H_{J_1J_2} = J_1\sum_{\langle ij\rangle} \mathbf S_i\cdot\mathbf S_j + J_2\sum_{\langle\langle ij\rangle\rangle} \mathbf S_i\cdot\mathbf S_j.
$$

Nearest neighbours prefer antiferromagnetic alignment when \\(J_1>0\\). Diagonal next-nearest neighbours also prefer antiferromagnetic alignment when \\(J_2>0\\). A spin cannot satisfy all these preferences simultaneously. That is frustration.

For small \\(J_2/J_1\\), Néel order dominates. For large \\(J_2/J_1\\), stripe or columnar order becomes favourable. The intermediate region is difficult and remains an active subject because competing states are close in energy.

This model also carries a sign problem for neural wavefunctions. At \\(J_2=0\\), a Marshall sign transformation supplies the exact sign structure on a bipartite lattice. Frustration destroys that simple global rule. The network must learn how interference changes across the phase diagram.

That is why my most revealing optimization failures occur here rather than in TFIM.

## One architecture does not imply one task

All three systems can be represented on a binary lattice and evaluated with local-energy ratios. Yet the learning burdens differ:

| Model | Dominant challenge |
|---|---|
| TFIM | quantum correlations across a phase transition |
| Rydberg | geometry-dependent blockade and density order |
| \\(J_1-J_2\\) | frustration, competing order, and nontrivial signs |

A benchmark spanning all three asks whether an architecture succeeds for a general reason or because it matches one model’s easiest structure.

## Conventions used in my calculations

Unless a run document says otherwise:

- lattice sites are numbered in row-major order;
- periodic pairs are enumerated once;
- TFIM uses \\(-JZZ-hX\\);
- the Heisenberg interaction means
  $$ \mathbf S_i\cdot\mathbf S_j = (X_iX_j+Y_iY_j+Z_iZ_j)/4 $$;
- Rydberg bit one is occupation;
- energies may be reported per site, but reference totals and densities are never mixed.

The equations are compact. Their conventions are part of the result.

## Why I keep all three

TFIM tells me whether the method can learn a clean quantum critical problem. Rydberg tests whether geometry and longer-range interactions are handled consistently. \\(J_1-J_2\\) asks the hardest question: can the model learn amplitudes and signs when the classical preferences themselves disagree?

Together they form something like an oral examination for a neural quantum state. Passing only one section is informative, but it is not yet a general education.

## Further reading

- [Sachdev, *Quantum Phase Transitions*](https://doi.org/10.1017/CBO9780511973765)
- [Bernien et al., *Probing many-body dynamics on a 51-atom quantum simulator*](https://doi.org/10.1038/nature24622)
- [Choo et al., *Symmetries and many-body excitations with neural-network quantum states*](https://doi.org/10.1103/PhysRevLett.121.167204)
- [Jiang et al., *Ground state of the spin-1/2 square-lattice J1-J2 Heisenberg model*](https://doi.org/10.1103/PhysRevB.86.024424)

