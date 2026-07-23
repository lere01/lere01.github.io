---
author: Faith O. Oyedemi
title: "A Quantum Simulation Is More Than Its Final Energy"
date: 2026-03-11
draft: true
tags:
- reproducibility
- scientific computing
- research practice
---

{{< katex >}}

A number such as

$$
e=-0.5861
$$

looks like a result. By itself, it is not.

Which Hamiltonian convention produced it? Was it a total energy or energy per site? Which boundary condition? Which seed? Was the checkpoint resumed? How many samples were used for the final estimate? Did the model carry a sign prior?

Scientific computing turns a mathematical question into a chain of executable decisions. Reproducibility means preserving that chain.

## Begin with a resolved experiment

I prefer each run to begin from a declarative document. It names:

- Hamiltonian and all couplings;
- lattice dimensions and boundaries;
- basis and site ordering;
- architecture and representation mode;
- parameter-budget target;
- optimizer and schedule;
- sample counts;
- seed;
- observables;
- checkpoint and reporting policy.

Templates are useful for avoiding repetition. The artifact must contain the fully resolved configuration after defaults and overrides have been applied.

Otherwise, changing a default can alter the meaning of an old command.

## Deterministic planning

The benchmark unit is a product of architecture, Hamiltonian, lattice, phase, budget, and seed. Expanding this grid should produce a deterministic ordered list of jobs and stable identifiers.

A job identifier should depend on the scientific configuration, not on the time at which it was launched. If I change \\(dt\\), sample count, or solver tolerance, the identifier should change. The altered run must not accidentally resume from an incompatible checkpoint.

This becomes crucial on SLURM or multi-GPU systems, where different workers execute disjoint portions of a plan.

## Checkpoints are scientific state

Saving model weights is not enough for an exact resume. Depending on the algorithm, a checkpoint may also require:

- optimizer or integrator state;
- random-number-generator states;
- current epoch or physical time;
- parameter-order fingerprint;
- resolved-config fingerprint;
- adaptive-controller history;
- sample-stream position;
- committed trajectory length;
- distributed world layout.

For stochastic dynamics, restoring weights while drawing an unrelated continuation is a new trajectory, not an exact resume.

I find it useful to think of a checkpoint as a transaction boundary. Everything before it is committed; everything after it can be reconstructed.

## Append-only observations

Long calculations fail. Machines restart, allocations expire, network storage pauses, and code meets inputs its author did not imagine.

Trajectory and attempt logs should therefore be append-only and recoverable. Accepted and rejected integration attempts should be separate records. Summaries can be regenerated from the primary log.

Atomic writes prevent a partially written checkpoint from masquerading as a valid one. Checksums can detect corruption in transferred artifacts.

This is less glamorous than a new architecture. It is also what allows an expensive trajectory to become evidence.

## Randomness needs a map

A single integer seed is useful but may not describe every random stream. Model initialization, training samples, evaluation samples, job ordering, and distributed ranks should derive deterministic child seeds from a documented scheme.

The rule should survive parallel execution. Rank zero and rank one must not accidentally produce identical samples, and changing the number of workers should not silently change the scientific job identity unless the algorithm requires the layout to remain fixed.

Determinism also has limits. GPU kernels and reduction order may produce small floating-point differences. Reproducibility should state whether it means:

- bitwise identical;
- numerically equivalent within tolerance;
- statistically consistent across repeated runs.

## Record failure, not only success

A benchmark that stores only completed runs suffers from survivorship bias. Divergence, out-of-memory errors, invalid parameter combinations, and watchdog terminations are part of architecture reliability.

Each planned job should end in a machine-readable state:

- completed;
- failed with a classified error;
- stopped intentionally;
- incomplete and resumable;
- excluded by preflight.

Preflight itself should construct the model and Hamiltonian, count parameters, estimate memory, probe the sampler, and select a resource-safe solver before a production allocation is consumed.

## Validate conventions across implementations

Independent implementations can agree on a spectrum while disagreeing on basis order. They can agree on a total energy while reporting different densities. They can both be Hermitian and still encode different detuning signs.

I use small reference vectors and matrix parity tests to seal:

- basis encoding;
- pair multiplicity;
- coupling signs;
- observable normalization;
- state transformations;
- serialization schemas.

The test fixture should be small enough to inspect and complete enough to fail when a scientific convention changes.

## What belongs beside a published figure

For every figure, I want a path back to:

1. raw run artifacts;
2. resolved configurations;
3. code revision;
4. analysis script;
5. reference data and provenance;
6. environment or dependency lock;
7. commands needed to regenerate the plot.

The plotting script should not contain undocumented corrections that are absent from the data pipeline. If a run is excluded, the criterion belongs in the analysis record.

## Communication is part of reproducibility

A physicist should not need to understand Python object graphs or Rust traits to reconstruct the calculation. Documentation should begin with the physical system and observable, then expose the smallest command or API needed to reproduce it.

Every symbol should have a convention. Every uncertainty should say what varies. Every approximation should have a convergence test.

That standard changes how I write code. If I cannot explain which state a sampler targets, I am not ready to optimize it. If I cannot describe what a checkpoint contains, I am not ready to call resume exact.

## The final number, restored

Now the energy density

$$
e=-0.5861
$$

can become meaningful. It belongs to a specified Hamiltonian, finite lattice, boundary condition, wavefunction family, optimization protocol, estimator, and uncertainty.

Reproducibility is not administrative work attached to the science. It is the structure that turns computation into a claim another person can examine.

## Further reading

- [National Academies, *Reproducibility and Replicability in Science*](https://doi.org/10.17226/25303)
- [Pineau et al., *Improving Reproducibility in Machine Learning Research*](https://arxiv.org/abs/2003.12206)
- [FAIR Guiding Principles for scientific data management](https://doi.org/10.1038/sdata.2016.18)
