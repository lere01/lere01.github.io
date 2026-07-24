"""Canonical CLI VMC study for an eight-site open Rydberg chain.

The model is trained through ``utilities.train`` using autoregressive samples
and local-energy gradients. Exact enumeration is used only after training to
validate each final state.

Run with:

    NQS_CLI_ROOT=~/Documents/cli python vmc_convergence.py
"""

from __future__ import annotations

import os
import shutil
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import torch

OUTPUT_DIR = Path(__file__).resolve().parent
CLI_ROOT = Path(
    os.environ.get("NQS_CLI_ROOT", Path.home() / "Documents" / "cli")
).expanduser()
sys.path.insert(0, str(CLI_ROOT))

from utilities import SchedulerCfg, VMCConfig, train
from src.hamiltonians.rydberg import RydbergHamiltonian
from src.models.interactions import compute_interactions
from src.models.mixer import MixerWaveFunction
from src.schemas.configs import RydbergConfig


N_SITES = 8
NUM_SAMPLES = 1000
MINI_BATCH_SIZE = 250
EPOCHS = 5000
SEEDS = (7, 19, 43)
TRAILING_WINDOW = 100
OMEGA = 1.0
DELTA = 1.2
RB = 3.0 ** (1.0 / 6.0)
EXACT_ENERGY_DENSITY = -0.776224936473
EXPERIMENT = "public_mixer_n8_canonical_vmc"
TRACKING_ROOT = Path("/private/tmp/mixer-n8-canonical-vmc")
ARTIFACT_ROOT = TRACKING_ROOT / "artifacts"


def basis_states(n_sites: int) -> np.ndarray:
    integers = np.arange(1 << n_sites, dtype=np.uint16)
    shifts = np.arange(n_sites, dtype=np.uint16)
    return ((integers[:, None] >> shifts[None, :]) & 1).astype(np.float64)


def exact_hamiltonian(states: np.ndarray) -> np.ndarray:
    n_states, n_sites = states.shape
    matrix = np.zeros((n_states, n_states), dtype=np.float64)
    distances = np.abs(np.arange(n_sites)[:, None] - np.arange(n_sites)[None, :])
    interactions = np.zeros_like(distances, dtype=np.float64)
    nonzero = distances > 0
    interactions[nonzero] = OMEGA * RB**6 / distances[nonzero].astype(float) ** 6
    diagonal = -DELTA * states.sum(axis=1)
    diagonal += 0.5 * np.einsum("bi,ij,bj->b", states, interactions, states)
    np.fill_diagonal(matrix, diagonal)
    for state_index in range(n_states):
        for site in range(n_sites):
            matrix[state_index, state_index ^ (1 << site)] = -0.5 * OMEGA
    return matrix


def make_model(seed: int) -> MixerWaveFunction:
    torch.manual_seed(seed)
    np.random.seed(seed)
    return MixerWaveFunction(
        Lx=8,
        Ly=1,
        patch_h=1,
        patch_w=1,
        d_model=32,
        num_layers=2,
        use_phase=False,
        device="cpu",
        mode="site",
        token_mlp_factor=1,
        channel_mlp_factor=2,
    )


def make_hamiltonian() -> RydbergHamiltonian:
    interactions = compute_interactions(8, 1, RB, OMEGA)
    return RydbergHamiltonian(
        RydbergConfig(
            delta=DELTA,
            Omega=OMEGA,
            interaction_matrix=interactions,
        )
    )


def make_config(seed: int) -> VMCConfig:
    return VMCConfig(
        Lx=8,
        Ly=1,
        Rb=RB,
        Omega=OMEGA,
        delta=DELTA,
        d_model=32,
        hidden_dim=32,
        num_layers=2,
        mode="site",
        patch_h=1,
        patch_w=1,
        num_samples=NUM_SAMPLES,
        mini_batch_size=MINI_BATCH_SIZE,
        epochs=EPOCHS,
        lr=1e-3,
        weight_decay=0.0,
        optimizer="adamw",
        scheduler_cfg=SchedulerCfg(
            scheduler_cls=torch.optim.lr_scheduler.MultiStepLR,
            kwargs={"milestones": [3500], "gamma": 0.2},
        ),
        grad_clip=1.0,
        clip_epochs=0,
        seed=seed,
        arch_type="mixer",
        hamil_type="rydberg",
        arch_name="MixerN8CanonicalVMC",
        db=f"sqlite:///{TRACKING_ROOT / f'mlflow-seed-{seed}.db'}",
        mlflow_artifact_root=str(ARTIFACT_ROOT / f"seed-{seed}"),
        experiment=EXPERIMENT,
        run_name=f"seed-{seed}",
        reference_energy_per_site=EXACT_ENERGY_DENSITY,
        compute_renyi2_final=False,
        sample_log_every=0,
        log_mag_x_every=0,
        mlflow_log_system_metrics=False,
        print_every=50,
    )


def metric_history(run_id: str, key: str) -> np.ndarray:
    client = mlflow.tracking.MlflowClient()
    history = sorted(client.get_metric_history(run_id, key), key=lambda item: item.step)
    return np.asarray([item.value for item in history], dtype=np.float64)


def find_run_id(seed: int) -> str:
    client = mlflow.tracking.MlflowClient()
    experiment = client.get_experiment_by_name(EXPERIMENT)
    if experiment is None:
        raise RuntimeError(f"MLflow experiment {EXPERIMENT!r} was not created")
    runs = client.search_runs(
        [experiment.experiment_id],
        filter_string=f"attributes.run_name = 'seed-{seed}'",
        order_by=["attributes.start_time DESC"],
        max_results=1,
    )
    if not runs:
        raise RuntimeError(f"MLflow run for seed {seed} was not found")
    return runs[0].info.run_id


def validate_final_state(
    model: MixerWaveFunction,
    states: torch.Tensor,
    hamiltonian: torch.Tensor,
    exact_state: torch.Tensor,
) -> dict[str, float]:
    model.eval()
    with torch.no_grad():
        psi = torch.exp(model.logpsi(states))
        psi = psi / torch.linalg.vector_norm(psi)
        h_psi = hamiltonian @ psi
        energy = torch.real(torch.vdot(psi, h_psi)).item()
        variance = (
            torch.real(torch.vdot(h_psi, h_psi)).item() - energy * energy
        )
        fidelity = torch.abs(torch.vdot(exact_state, psi)).square().item()
        probabilities = psi.abs().square()
        occupation = (
            probabilities * states.to(torch.float32).mean(dim=1)
        ).sum().item()
    return {
        "exact_eval_energy_density": energy / N_SITES,
        "exact_eval_variance_density": max(variance, 0.0) / N_SITES,
        "fidelity": fidelity,
        "occupation": occupation,
    }


def write_training_data(results: list[dict[str, object]]) -> None:
    path = OUTPUT_DIR / "vmc-training-data.txt"
    with path.open("w", encoding="utf-8") as handle:
        handle.write("# Canonical CLI VMC trajectories for the N=8 Mixer study\n")
        handle.write(
            "# Each epoch draws 1000 autoregressive samples and applies four AdamW updates\n"
        )
        handle.write(
            "# columns: epoch, then energy_density, energy_density_stderr, "
            "sample_variance_density for seeds 7, 19, 43\n"
        )
        for epoch in range(EPOCHS):
            values = [str(epoch + 1)]
            for result in results:
                values.extend(
                    f"{float(result[key][epoch]):.12e}"
                    for key in ("energy", "stderr", "variance")
                )
            handle.write(" ".join(values) + "\n")


def write_summary(results: list[dict[str, object]]) -> None:
    path = OUTPUT_DIR / "vmc-seed-summary.txt"
    trailing_means = []
    with path.open("w", encoding="utf-8") as handle:
        handle.write(
            "# seed trailing_100_vmc_mean trailing_100_vmc_std "
            "raw_final_epoch exact_eval_energy_density "
            "exact_eval_variance_density fidelity occupation\n"
        )
        for result in results:
            energy = np.asarray(result["energy"])
            mean = float(energy[-TRAILING_WINDOW:].mean())
            std = float(energy[-TRAILING_WINDOW:].std(ddof=1))
            trailing_means.append(mean)
            validation = result["validation"]
            handle.write(
                f"{result['seed']} {mean:.12e} {std:.12e} "
                f"{energy[-1]:.12e} "
                f"{validation['exact_eval_energy_density']:.12e} "
                f"{validation['exact_eval_variance_density']:.12e} "
                f"{validation['fidelity']:.12e} "
                f"{validation['occupation']:.12e}\n"
            )
        means = np.asarray(trailing_means)
        handle.write(
            f"# across_seed_mean={means.mean():.12e}\n"
            f"# across_seed_std={means.std(ddof=1):.12e}\n"
            f"# across_seed_sem={means.std(ddof=1) / np.sqrt(len(means)):.12e}\n"
        )


def moving_average(values: np.ndarray, window: int) -> np.ndarray:
    output = np.full(values.shape, np.nan, dtype=np.float64)
    output[window - 1 :] = np.convolve(
        values, np.ones(window) / window, mode="valid"
    )
    return output


def plot_results(results: list[dict[str, object]]) -> None:
    colors = ("#2563eb", "#c2410c", "#047857")
    epochs = np.arange(1, EPOCHS + 1)
    plt.rcParams.update(
        {
            "font.size": 12,
            "axes.titlesize": 15,
            "axes.labelsize": 13,
            "legend.fontsize": 10,
        }
    )

    figure, axis = plt.subplots(figsize=(9.5, 5.4), constrained_layout=True)
    for result, color in zip(results, colors):
        energy = np.asarray(result["energy"])
        average = moving_average(energy, 25)
        axis.plot(epochs, energy, color=color, alpha=0.16, linewidth=0.7)
        axis.plot(
            epochs,
            average,
            color=color,
            linewidth=2,
            label=f"seed {result['seed']} (25-epoch mean)",
        )
    axis.axhline(
        EXACT_ENERGY_DENSITY,
        color="#292524",
        linestyle="--",
        linewidth=1.8,
        label=f"exact: {EXACT_ENERGY_DENSITY:.6f}",
    )
    axis.axvline(3500, color="#292524", linestyle=":", linewidth=1.2)
    axis.set(
        title="Canonical VMC convergence from three independent seeds",
        xlabel="VMC epoch (1,000 samples and four AdamW updates)",
        ylabel=r"Sampled energy per atom, $E/N$",
    )
    axis.set_ylim(-0.79, -0.60)
    axis.grid(True, alpha=0.2)
    axis.legend(loc="lower right")
    figure.savefig(OUTPUT_DIR / "vmc-energy-convergence.png", dpi=180)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(9.5, 5.4))
    for result, color in zip(results, colors):
        energy = np.asarray(result["energy"])
        error = np.abs(moving_average(energy, 25) - EXACT_ENERGY_DENSITY)
        axis.semilogy(
            epochs,
            error,
            color=color,
            linewidth=2,
            label=f"seed {result['seed']}",
        )
    axis.axvline(
        3500,
        color="#292524",
        linestyle=":",
        linewidth=1.2,
        label="learning rate reduced",
    )
    axis.set(
        title="Energy-density error during stochastic VMC training",
        xlabel="VMC epoch",
        ylabel=r"$|E_{\mathrm{VMC}}/N-E_0/N|$ (25-epoch mean)",
    )
    axis.grid(True, which="both", alpha=0.2)
    axis.legend()
    figure.tight_layout()
    figure.savefig(OUTPUT_DIR / "vmc-energy-error.png", dpi=180)
    plt.close(figure)


def run_seed(seed: int) -> dict[str, object]:
    torch.set_num_threads(1)
    states_np = basis_states(N_SITES)
    matrix_np = exact_hamiltonian(states_np)
    eigenvalues, eigenvectors = np.linalg.eigh(matrix_np)
    exact_energy = float(eigenvalues[0])
    if not np.isclose(exact_energy / N_SITES, EXACT_ENERGY_DENSITY, atol=1e-10):
        raise RuntimeError("Exact reference changed unexpectedly")

    model = make_model(seed)
    output = train(
        model,
        make_hamiltonian(),
        cfg=make_config(seed),
        device="cpu",
    )
    run_id = find_run_id(seed)
    energy = metric_history(run_id, "energy_density")
    stderr = metric_history(run_id, "energy_density_stderr")
    variance = metric_history(run_id, "energy_variance_density")
    if not (len(energy) == len(stderr) == len(variance) == EPOCHS):
        raise RuntimeError(f"Incomplete metric history for seed {seed}")

    return {
        "seed": seed,
        "energy": energy,
        "stderr": stderr,
        "variance": variance,
        "validation": validate_final_state(
            output.model,
            torch.tensor(states_np, dtype=torch.long),
            torch.tensor(matrix_np, dtype=torch.complex64),
            torch.tensor(eigenvectors[:, 0], dtype=torch.complex64),
        ),
    }


def load_training_data() -> list[dict[str, object]]:
    table = np.loadtxt(OUTPUT_DIR / "vmc-training-data.txt")
    results = []
    for seed_index, seed in enumerate(SEEDS):
        start = 1 + 3 * seed_index
        results.append(
            {
                "seed": seed,
                "energy": table[:, start],
                "stderr": table[:, start + 1],
                "variance": table[:, start + 2],
            }
        )
    return results


def main() -> None:
    if "--plots-only" in sys.argv:
        plot_results(load_training_data())
        return

    if TRACKING_ROOT.exists():
        shutil.rmtree(TRACKING_ROOT)
    ARTIFACT_ROOT.mkdir(parents=True, exist_ok=True)

    with ProcessPoolExecutor(max_workers=len(SEEDS)) as executor:
        results = list(executor.map(run_seed, SEEDS))

    write_training_data(results)
    write_summary(results)
    plot_results(results)
    print((OUTPUT_DIR / "vmc-seed-summary.txt").read_text())


if __name__ == "__main__":
    main()
