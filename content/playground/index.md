---
menu: "main"
date: "2025-11-18"
weight: 30
---

<style>
  #run-button {
    background-color: #111827; /* dark neutral */
    color: #f9fafb;            /* near-white */
    padding: 0.4rem 0.9rem;
    border-radius: 0.375rem;
    border: none;
    font-size: 0.875rem;
    font-weight: 500;
    cursor: pointer;
  }

  #run-button:hover:not(:disabled) {
    background-color: #1f2937;
  }

  #run-button:disabled {
    opacity: 0.5;
    cursor: not-allowed;
  }

</style>
<link
  rel="stylesheet"
  href="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/katex.min.css"
/>

<script
  defer
  src="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/katex.min.js"
></script>

<script
  defer
  src="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/contrib/auto-render.min.js"
></script>

<script>
  document.addEventListener("DOMContentLoaded", function () {
    if (typeof renderMathInElement !== "function") return;

    var root = document.getElementById("nqs-sampler") || document.body;

    renderMathInElement(root, {
      delimiters: [
        { left: "$$", right: "$$", display: true },
        { left: "$", right: "$", display: false },
        { left: "\\(", right: "\\)", display: false },
        { left: "\\[", right: "\\]", display: true }
      ],
      throwOnError: false
    });
  });
</script>

<div
  id="nqs-sampler"
  class="max-w-2xl mx-auto my-8 space-y-6"
>
  <!-- Intro -->
  <section class="space-y-3">
    <h1 class="text-base font-semibold text-neutral-900">
      Sampling Playground
    </h1>
    <p class="text-sm text-neutral-500">
      This page provides an interactive interface for sampling from pretrained neural quantum state (NQS) models. Each model represents an approximate ground-state wavefunction for a particular quantum spin Hamiltonian (for example, 2D TFIM, Rydberg, or J1–J2) on a given lattice size.
    </p>
    <p class="text-sm text-neutral-500">
      When you select a model and choose the number of samples, the backend loads the corresponding wavefunction, and draws Monte Carlo samples from the distribution $|\psi(\sigma)|^2$. A small preview of the first few configurations is shown below; the full set of samples and observables is available for download as JSON or CSV.
    </p>
    <div class="text-[11px] text-neutral-500 space-y-1">
      <p class="font-medium text-neutral-600">
        How to read model names
      </p>
      <p>
        Model labels follow the pattern:
        <span class="font-mono">Hamiltonian / LatticeSize – Tag (Architecture)</span>.
      </p>
      <ul class="list-disc list-inside space-y-0.5">
        <li><span class="font-semibold">Hamiltonian</span>: e.g. 2DTFIM, Rydberg, J1J2.</li>
        <li><span class="font-semibold">LatticeSize</span>: e.g. <span class="font-mono">4x4</span>, <span class="font-mono">8x8</span>, indicating the spin lattice dimensions.</li>
        <li><span class="font-semibold">Tag</span>: distinguishes parameter regimes or training setups (e.g. detuning, phase, baseline, fine-tuned).</li>
        <li><span class="font-semibold">Architecture</span>: the neural ansatz used, such as Transformer, or RNN.</li>
      </ul>
      <p>
        For example,
        <span class="font-mono">TFIM 4x4 – h_0.5 (prnn)</span>
        denotes a 4x4 2d TFIM model trained at J=1, h = 0.5 using a Patched RNN wavefunction.
      </p>
    </div>
  </section>

  <!-- Controls -->
  <section
    id="nqs-sampler-controls"
    class="space-y-4 border border-neutral-200 rounded-lg p-4 bg-neutral-50"
  >
    <form id="sampler-form" class="space-y-4">
      <!-- Model selector -->
      <div class="space-y-1">
        <label
          class="block text-sm font-medium mb-1"
          for="model_key"
        >
          Model
        </label>
        <select
          id="model_key"
          class="w-full border rounded-md px-2 py-1 text-sm"
        >
          <option value="" disabled selected>Loading models...</option>
        </select>
        <p class="text-[11px] text-neutral-500 mt-1">
          Models are grouped by Hamiltonian, system size, tag, and architecture.
        </p>
      </div>

<!-- Sampling params -->
<div class="grid grid-cols-1 sm:grid-cols-2 gap-4">
        <div class="space-y-1">
          <label
            class="block text-sm font-medium mb-1"
            for="num_samples"
          >
            Number of samples
          </label>
          <input
            type="number"
            id="num_samples"
            value="1000"
            min="1"
            max="100000"
            class="w-full border rounded-md px-2 py-1 text-sm"
          />
          <p class="text-[11px] text-neutral-500">
            Larger values give smoother estimates but take longer to compute.
          </p>
        </div>
      </div>

<!-- Actions -->
<div class="flex flex-wrap items-center gap-2">
        <button
          type="submit"
          id="run-button"
          class="inline-flex items-center px-3 py-1.5 text-sm font-medium rounded-md bg-neutral-900 text-neutral-50 hover:bg-neutral-800 disabled:opacity-50"
        >
          Run
        </button>
        <span
          id="status-text"
          class="text-xs text-neutral-500"
        ></span>
      </div>
    </form>

<!-- Error message -->
<div
      id="error-box"
      class="hidden text-sm text-red-600 border border-red-200 bg-red-50 px-3 py-2 rounded-md mt-2"
    ></div>
  </section>

<!-- Results -->
  <section
    id="results-card"
    class="hidden border border-neutral-200 rounded-lg p-4 bg-neutral-50 space-y-3"
  >
    <h2 class="text-sm font-semibold text-neutral-800">
      Sampling result
    </h2>

<!-- Summary -->
<div class="grid grid-cols-2 gap-3 text-sm">
      <div>
        <div class="text-xs text-neutral-500">Model</div>
        <div
          id="summary-model"
          class="font-mono text-xs break-all"
        ></div>
      </div>

<div>
<div class="text-xs text-neutral-500">
          Number of samples
        </div>
        <div
          id="summary-num-samples"
          class="font-mono text-xs"
        ></div>
      </div>

<div>
        <div class="text-xs text-neutral-500">Energy</div>
        <div
          id="summary-energy"
          class="font-mono text-xs"
        ></div>
      </div>

<div>
        <div class="text-xs text-neutral-500">Variance</div>
        <div
          id="summary-variance"
          class="font-mono text-xs"
        ></div>
      </div>
    </div>

<!-- Preview table -->
<div class="mt-2 space-y-1">
      <div class="flex items-center justify-between">
        <span class="text-xs text-neutral-500">
          Preview of first
          <span id="preview-count">0</span>
          samples
        </span>
      </div>

<div class="border border-neutral-200 rounded-md overflow-x-auto max-h-64">
        <table class="min-w-full text-xs">
          <thead class="bg-neutral-100 sticky top-0">
            <tr>
              <th
                class="px-2 py-1 text-left font-medium text-neutral-600"
              >
                #
              </th>
              <th
                class="px-2 py-1 text-left font-medium text-neutral-600"
              >
                Configuration
              </th>
            </tr>
          </thead>
          <tbody
            id="preview-tbody"
            class="divide-y divide-neutral-200"
          ></tbody>
        </table>
      </div>
    </div>

<!-- Download buttons -->
<div
      class="flex flex-col sm:flex-row sm:items-center gap-3 pt-2 border-t border-neutral-200"
    >
      <div class="flex items-center gap-2">
        <button
          id="download-json"
          class="px-3 py-1.5 text-xs rounded-md border border-neutral-300 bg-white hover:bg-neutral-100 disabled:opacity-50"
          type="button"
          disabled
        >
          Download samples (JSON)
        </button>

<button
          id="download-csv"
          class="px-3 py-1.5 text-xs rounded-md border border-neutral-300 bg-white hover:bg-neutral-100 disabled:opacity-50"
          type="button"
          disabled
        >
          Download samples (CSV)
        </button>
      </div>

<span class="text-[11px] text-neutral-400">
        Full samples are available via download. Only a small preview is rendered in the browser.
      </span>
    </div>
  </section>
</div>

<script>
  (function () {
    "use strict";

    // --- Configuration / "props" -------------------------------------------
    const API_BASE = "https://oakor2024-nqs-sampler.hf.space";

    // --- Component state ----------------------------------------------------
    const state = {
      models: [],
      lastResult: null,
    };

    // --- Element lookups ("bindings") --------------------------------------
    const form = document.getElementById("sampler-form");
    const modelSelect = document.getElementById("model_key");
    const runButton = document.getElementById("run-button");
    const statusText = document.getElementById("status-text");

    const errorBox = document.getElementById("error-box");
    const resultsCard = document.getElementById("results-card");

    const summaryModel = document.getElementById("summary-model");
    const summaryNumSamples = document.getElementById("summary-num-samples");
    const summaryEnergy = document.getElementById("summary-energy");
    const summaryVariance = document.getElementById("summary-variance");
    const previewCountEl = document.getElementById("preview-count");
    const previewTbody = document.getElementById("preview-tbody");

    const downloadJsonBtn = document.getElementById("download-json");
    const downloadCsvBtn = document.getElementById("download-csv");

    // --- Guard: if container not present, bail out -------------------------
    if (!form || !modelSelect) {
      console.warn(
        "[NQS Sampler] Required DOM elements not found; aborting initialisation."
      );
      return;
    }

    // --- Small helpers ("actions") -----------------------------------------
    function setLoading(isLoading, message) {
      runButton.disabled = isLoading;
      statusText.textContent = message || "";
    }

    function showError(message) {
      errorBox.textContent = message;
      errorBox.classList.remove("hidden");
    }

    function clearError() {
      errorBox.textContent = "";
      errorBox.classList.add("hidden");
    }

    function enableDownloads(enabled) {
      if (!downloadJsonBtn || !downloadCsvBtn) return;
      downloadJsonBtn.disabled = !enabled;
      downloadCsvBtn.disabled = !enabled;
    }

    function downloadBlob(content, filename, contentType) {
      const blob = new Blob([content], { type: contentType });
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = filename;
      document.body.appendChild(a);
      a.click();
      a.remove();
      URL.revokeObjectURL(url);
    }

    // --- Data fetching ------------------------------------------------------
    async function fetchModels() {
      const res = await fetch(API_BASE + "/models");
      if (!res.ok) {
        throw new Error("Failed to load models (status " + res.status + ")");
      }
      return res.json();
    }

    async function fetchSamples(modelId, numSamples) {
      const res = await fetch(API_BASE + "/sample", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          model_id: modelId,
          num_samples: numSamples,
        }),
      });

      if (!res.ok) {
        let detail = "Request failed with status " + res.status;
        try {
          const body = await res.json();
          if (body && body.detail) detail = body.detail;
        } catch (e) {
          // ignore JSON parse errors
        }
        throw new Error(detail);
      }

      return res.json();
    }

    // --- Renderers ---------------------------------------------------------
    function renderModelOptions(models) {
      modelSelect.innerHTML = "";

      if (!Array.isArray(models) || models.length === 0) {
        const opt = document.createElement("option");
        opt.value = "";
        opt.disabled = true;
        opt.selected = true;
        opt.textContent = "No models available";
        modelSelect.appendChild(opt);
        return;
      }

      models.forEach((m) => {
        // Backend shape: { key: { id, hamiltonian, system_size, tag, architecture, ... } }
        const k = m.key || m;
        const id = k.id;
        if (!id) return;

        const parts = [];
        if (k.hamiltonian) parts.push(k.hamiltonian);
        if (k.system_size) parts.push(k.system_size);
        if (k.tag) parts.push("– " + k.tag);
        if (k.architecture) parts.push("(" + k.architecture + ")");

        const label = parts.join(" ") || id;
        const opt = document.createElement("option");
        opt.value = id;
        opt.textContent = label;
        modelSelect.appendChild(opt);
      });
    }

    function renderResult(result, modelId, numSamples) {
      const obs = result.observables || {};
      const samples = result.samples || [];
      const preview = samples.slice(0, 10);

      // Summary
      summaryModel.textContent = modelId;
      summaryNumSamples.textContent = String(numSamples);
      summaryEnergy.textContent =
        obs.energy !== undefined && obs.energy !== null
          ? String(obs.energy)
          : "N/A";
      summaryVariance.textContent =
        obs.variance !== undefined && obs.variance !== null
          ? String(obs.variance)
          : "N/A";

      // Preview
      previewTbody.innerHTML = "";
      previewCountEl.textContent = String(preview.length);

      preview.forEach((row, idx) => {
        const tr = document.createElement("tr");

        const idxTd = document.createElement("td");
        idxTd.className = "px-2 py-1 align-top text-neutral-500";
        idxTd.textContent = String(idx);
        tr.appendChild(idxTd);

        const cfgTd = document.createElement("td");
        cfgTd.className = "px-2 py-1 font-mono text-[11px]";
        cfgTd.textContent = Array.isArray(row) ? row.join(" ") : String(row);
        tr.appendChild(cfgTd);

        previewTbody.appendChild(tr);
      });

      resultsCard.classList.remove("hidden");
      enableDownloads(samples.length > 0);
    }

    // --- Event handlers -----------------------------------------------------
    async function handleLoadModels() {
      clearError();
      setLoading(true, "Loading models...");
      try {
        const models = await fetchModels();
        state.models = models;
        renderModelOptions(models);
        setLoading(false, "");
      } catch (err) {
        console.error(err);
        renderModelOptions([]);
        setLoading(false, "");
        showError("Error loading models. Check console for details.");
      }
    }

    async function handleSubmit(event) {
      event.preventDefault();
      clearError();
      resultsCard.classList.add("hidden");
      state.lastResult = null;
      enableDownloads(false);

      const modelId = modelSelect.value;
      if (!modelId) {
        showError("Please select a model.");
        return;
      }

      const numSamples = parseInt(
        document.getElementById("num_samples").value,
        10
      );

      if (!Number.isFinite(numSamples) || numSamples <= 0) {
        showError("Please enter a valid number of samples.");
        return;
      }

      setLoading(true, "Running sampling...");
      try {
        const result = await fetchSamples(modelId, numSamples);
        state.lastResult = result;
        renderResult(result, modelId, numSamples);
        setLoading(false, "Done.");
      } catch (err) {
        console.error(err);
        showError("Error: " + err.message);
        setLoading(false, "");
      }
    }

    function handleDownloadJson() {
      if (!state.lastResult || !Array.isArray(state.lastResult.samples)) {
        return;
      }
      const payload = {
        model_id: state.lastResult.model_id || state.lastResult.model_key,
        num_samples: state.lastResult.num_samples,
        samples: state.lastResult.samples,
        observables: state.lastResult.observables || null,
        metadata: state.lastResult.metadata || null,
      };
      const json = JSON.stringify(payload, null, 2);
      downloadBlob(json, "nqs_samples.json", "application/json");
    }

    function handleDownloadCsv() {
      if (!state.lastResult || !Array.isArray(state.lastResult.samples)) {
        return;
      }

      const samples = state.lastResult.samples;
      if (!samples.length) return;

      const N = samples[0].length ?? 0;
      const header =
        "sample_index," +
        Array.from({ length: N }, function (_, i) {
          return "s" + i;
        }).join(",");

      const lines = [header];

      samples.forEach(function (row, idx) {
        const flat = Array.isArray(row) ? row : [row];
        const line =
          String(idx) +
          "," +
          flat
            .map(function (v) {
              return v === null || v === undefined ? "" : String(v);
            })
            .join(",");
        lines.push(line);
      });

      const csv = lines.join("\n");
      downloadBlob(csv, "nqs_samples.csv", "text/csv");
    }

    // --- Wiring ("onMount") -------------------------------------------------
    function init() {
      form.addEventListener("submit", handleSubmit);
      if (downloadJsonBtn) {
        downloadJsonBtn.addEventListener("click", handleDownloadJson);
      }
      if (downloadCsvBtn) {
        downloadCsvBtn.addEventListener("click", handleDownloadCsv);
      }
      handleLoadModels();
    }

    if (document.readyState === "loading") {
      document.addEventListener("DOMContentLoaded", init);
    } else {
      init();
    }
  })();
</script>
