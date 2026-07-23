---
author: Faith O. Oyedemi
title: "Why Signs Become the Hard Part of a Frustrated Wavefunction"
date: 2025-12-31
draft: true
tags:
- sign structure
- frustrated magnetism
- J1-J2 model
---

{{< katex >}}

Imagine that I give a neural network the exact probability of every spin configuration. Has it learned the quantum state?

Not necessarily.

The missing information is the relative sign or phase. In an unfrustrated model that information may follow a simple rule. In a frustrated magnet it can become the part that refuses to generalize.

## Same probabilities, different physics

Write a real wavefunction as

$$
\psi(\sigma)=s(\sigma)\sqrt{p(\sigma)}, \qquad s(\sigma)\in\{-1,+1\}.
$$

Changing \\(s(\sigma)\\) leaves the Born probability \\(p(\sigma)\\) untouched. Diagonal observables in the computational basis are also unchanged. Off-diagonal energy terms are not.

For connected configurations \\(\sigma\\) and \\(\sigma'\\), the local energy contains

$$
H_{\sigma\sigma'} \frac{\psi(\sigma')}{\psi(\sigma)} = H_{\sigma\sigma'} \frac{s(\sigma')}{s(\sigma)} \sqrt{\frac{p(\sigma')}{p(\sigma)}}.
$$

The sign ratio decides whether the contribution interferes constructively or destructively. A state can reproduce the amplitude distribution reasonably well and still miss the ground-state energy because these ratios are wrong.

## The Marshall sign rule

For the antiferromagnetic Heisenberg model on a bipartite lattice, a basis transformation due to Marshall supplies a simple sign pattern. If \\(A\\) is one sublattice, one conventional form is

$$
s_{\mathrm M}(\sigma)=(-1)^{N_{\downarrow,A}},
$$

where \\(N_{\downarrow,A}\\) counts down spins on sublattice \\(A\\). Equivalent expressions depend on basis conventions.

Applying the rule is not merely a numerical trick. It incorporates known physics into the ansatz and converts a difficult sign pattern into a simpler one.

The trouble begins when next-nearest-neighbour \\(J_2\\) bonds connect sites on the same sublattice. The problem is no longer bipartite in the relevant sense. Marshall’s rule remains a useful prior in part of the phase diagram, but it is no longer exact everywhere.

## Expressibility is not the same as generalization

[Westerhout and collaborators](https://doi.org/10.1038/s41467-020-15402-w) studied neural approximations of frustrated ground states and separated amplitude learning from sign learning. Their results showed that sign generalization deteriorated sharply in frustrated regimes even when the architecture could express the state well under more favourable training information.

That distinction changed how I interpret a failed run:

- **Expressibility:** Does some parameter setting represent the target?
- **Optimization:** Can the training procedure find it?
- **Generalization:** Do the signs inferred from sampled configurations extend to important configurations not effectively learned?

A larger model attacks only part of this triangle.

## The race between amplitude and sign

Variational training couples the two. Suppose a configuration currently carries a harmful sign. The optimizer has at least two ways to reduce its energy contribution:

1. correct the sign;
2. suppress the amplitude so the configuration is rarely sampled.

The second route can be easier. Once the amplitude becomes tiny, the model sees that configuration less often, so the signal needed to repair its phase also weakens. Amplitude learning can outrun sign learning and close the door behind itself.

This is the mechanism I mean by an amplitude-sign race. It explains why a model may collapse onto a small classical support even though its nominal capacity is large.

## Priors help, but they change the question

If I supply a Marshall or columnar gauge, I am asking:

> Can the network learn the remaining correction around this physical sign prior?

Without it, I ask:

> Can the network discover the sign structure from the variational signal alone?

Both are legitimate scientific questions. Their results should not be placed in the same table without a label.

In my experiments, a discrete gauge switch across the \\(J_1-J_2\\) diagram can be highly effective. It is also manual knowledge of where one sign pattern should give way to another. A prior-free model would be more satisfying, but the evidence so far says that this is an optimization problem, not merely a request for more parameters.

## How should signs be measured?

For a small lattice with an exact reference \\(\psi_0\\), I can compute weighted sign agreement:

$$
A_{\mathrm{sign}} = \sum_\sigma |\psi_0(\sigma)|^2\, \mathbf 1[ s_\theta(\sigma)=s_0(\sigma) ].
$$

Weighting matters. A wrong sign on a configuration with negligible exact probability should not count as much as a wrong sign on a dominant configuration.

The full fidelity is stricter:

$$
F=|\langle\psi_0|\psi_\theta\rangle|^2.
$$

For larger systems, exact vectors are unavailable. Symmetry weights, total-spin observables, structure factors, and consistency across sizes become indirect diagnostics. Energy alone may not reveal the mechanism of failure.

## A curious simplification

There is evidence that sign structures themselves need not form a glassy optimization landscape once amplitudes are known. [Westerhout et al.](https://doi.org/10.1038/s42005-023-01388-6) mapped sign reconstruction to an auxiliary classical Ising problem and recovered signs for frustrated models.

This suggests a provocative interpretation: the hard part may be the *coupled* learning dynamics of amplitude and sign, not an intrinsically impossible sign landscape.

That leaves several routes open:

- update phase parameters more aggressively than amplitude parameters;
- pretrain phases with a suitable phase architecture;
- learn amplitudes and signs in separate stages;
- include symmetry-aware inputs without fixing the output sign;
- start several seeds and select basins with early diagnostics.

These are research directions, not conclusions. My current conclusion is narrower: in frustrated magnets, a good probability model is not yet a good wavefunction, and optimization can hide sign errors by erasing the configurations that expose them.

## Further reading

- [Marshall, *Antiferromagnetism*](https://doi.org/10.1098/rspa.1955.0108)
- [Westerhout et al., *Generalization properties of neural network approximations to frustrated magnet ground states*](https://doi.org/10.1038/s41467-020-15402-w)
- [Szabó and Castelnovo, *Neural network wave functions and the sign problem*](https://doi.org/10.1103/PhysRevResearch.2.033075)
- [Westerhout et al., *Many-body quantum sign structures as non-glassy Ising models*](https://doi.org/10.1038/s42005-023-01388-6)

