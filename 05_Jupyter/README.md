# JupyterLab projects: inspect, train, compare, explain

Start with [05: Running Jupyter Lab](05_Running_Jupyter_Lab.ipynb), then choose a project below. The introduction retains its explicit deployment/screenshot and live stop/restore TODOs. The follow-on lessons contain executable Python cells and bounded offline-first examples. They do not require the endpoint labs, a GPU, or gateway credentials unless you explicitly choose an optional live feature.

## Pick a lesson

| Lab | Outcome | Default and optional data/access |
|---|---|---|
| [05b: Explore MNIST](05b_Explore_MNIST.ipynb) | Inspect dimensions, pixel ranges, examples and label balance before training | Official MNIST opt-in; named 8×8 digits fallback offline |
| [05c: Train a Digit Classifier](05c_Train_a_Digit_Classifier.ipynb) | CPU classifier, held-out confusion matrix, error examples and safe numeric checkpoint | Same bounded image data; optional compatible PyTorch GPU extension |
| [05d: Explore EMNIST](05d_Explore_EMNIST.ipynb) | Letters/digits, orientation and correct label mappings | Synthetic contract demonstration offline; explicit large NIST archive opt-in or pre-staged cache |
| [05e: Compare Experiments](05e_Compare_Experiments.ipynb) | Train/validation/test discipline, three candidates, local CSV/JSON | Optional explicitly enabled ClearML metrics logging |
| [05f: Draw a Digit](05f_Draw_a_Digit.ipynb) | Interactive pixel drawing and an always-available NumPy fallback | Local model and inputs; optional ipywidgets frontend |
| [05g: Data Quality and Robustness](05g_Test_Data_Quality_and_Robustness.ipynb) | Label mistakes, missing pixels, imbalance, and controlled noise sensitivity | Bundled digits; copied corruptions never overwrite source arrays |
| [05h: Ask Public Documents](05h_Ask_Public_Documents.ipynb) | Bounded TF-IDF retrieval, cited extractive answers and abstention | Two allowlisted local READMEs; optional governed gateway answer |
| [05i: Benchmark CPU and GPU](05i_Benchmark_CPU_and_GPU.ipynb) | Warmed repeated timing, transfer cost and CUDA synchronization | NumPy CPU; compatible PyTorch and explicitly opted-in CUDA comparison |
| [05j: Create an Experiment Report](05j_Create_an_Experiment_Report.ipynb) | Reproducible Markdown report, metrics JSON/CSV and plot | Standalone offline digits run, local artifacts |
| [05k: Forecast Bike Demand](05k_Forecast_Bike_Demand.ipynb) | Chronological train/validation/test split and leakage-aware features | Synthetic seasonal data; optional small UCI Capital Bikeshare archive |
| [05l: Detect Telemetry Anomalies](05l_Detect_Telemetry_Anomalies.ipynb) | Train-normal detector, validation threshold, precision/recall and false positives | Seeded invented telemetry, no real operational data |
| [05m: Fashion Similarity](05m_Explore_Fashion_Similarity.ipynb) | PCA, nearest neighbors, clusters and interpretation | Official Fashion-MNIST opt-in; honestly named digit-method fallback |
| [05n: Clean Messy Data](05n_Clean_Messy_Data.ipynb) | Typed columns, quarantine, duplicates, UTC, pandas/SQLite agreement | Embedded invented CSV, in-memory SQL |

Suggested mini-paths:
- Image ML: **05b → 05c → 05e → 05f → 05g → 05j**. Add **05d** for letters or **05m** for clothing.
- Data practice: **05n → 05k → 05l → 05j**.
- Retrieval or hardware: choose **05h** or **05i** independently.

Each notebook starts from scratch. It does not depend on hidden state or outputs from a previous lesson. Read its data/source banner: a fallback is a different dataset, never a substitute benchmark claim.

## Prepare one approved Python environment

Use Python 3.10+ in the hosted JupyterLab environment or another approved local environment. In that environment's terminal, from the repository root:

```bash
python -m pip install -r 05_Jupyter/requirements.txt
```

Select the corresponding kernel. JupyterLab itself must already be available in the approved image; the introduction covers launch and kernel selection. No notebook installs packages automatically. If public package hosts are unavailable, have the administrator prepare the image or approved package mirror.

Optional capabilities are intentionally not forced into the CPU requirements:
- **Interactive drawing:** install `ipywidgets>=8.1,<9` through the approved source and enable the JupyterLab widget frontend. The NumPy fallback still works without it.
- **EMNIST / neural-network extension / PyTorch timing:** use the [official PyTorch installer guidance](https://pytorch.org/get-started/locally/) for a compatible approved `torch`/`torchvision` pair. Do not copy a CUDA wheel command from an unrelated machine. The notebook does not allocate a GPU for you.
- **ClearML:** optional `clearml>=1.16,<3` and approved server/project credentials supplied outside notebooks. Leave logging off unless the selected metrics may be sent there. The lesson disables automatic framework/argument/resource instrumentation and never uploads model files.
- **Live RAG generation:** Section 02 profile/alias permissions and the [VS Code gateway client's requirements](../07_VSCode/requirements.txt). Read 05h's explicit one-call opt-in and data-sharing warning. The default extractive path uses no LLM or credentials.

See [DATASETS.md](DATASETS.md) for attribution, separate dataset licenses, exact download flags, cache layout, data-size warnings, and interpretation limits. EMNIST's roughly 562 MB archive requires a separate explicit large-download opt-in and several GB of free disk when expanded. MNIST and Fashion-MNIST are much smaller. Never bypass a blocked host or TLS warning.

## Save useful results, not secrets

Notebook files contain code and saved outputs. Kernels contain transient state. Checkpoints contain a model, and workspace snapshots contain files. These are different things. Save source and useful artifacts in the confirmed persistent workspace, record package versions, then stop the managed session through its controls and verify it stopped.

Generated data, images, reports, and numeric checkpoints go under ignored `lab_outputs/`. The image checkpoint loader accepts numeric arrays with `allow_pickle=False`; it does not load arbitrary pickled Python objects. Never commit caches, `.env` files, tokens, or private notebook outputs. Ignored secrets can still enter a snapshot, and an optional tracking SDK can send metadata to its configured service.

## Validate offline

For developer validation, install the section test/kernel requirements in the environment where you run checks:

```bash
python -m pip install -r 05_Jupyter/requirements-dev.txt
python validate_course.py
python -m unittest discover -s 05_Jupyter -p 'test_*.py'
python 05_Jupyter/validate_notebooks.py
```

`validate_notebooks.py` copies this repository's lesson sources and selected documentation into a temporary directory, blocks Python-level outbound HTTP/TCP connection attempts inside each execution process, and executes all 13 follow-on notebooks in separate fresh Python processes. It removes image data caches so default offline fallbacks are exercised. This is an offline regression guard, not a security sandbox. Generated outputs stay in the temporary copy, not committed notebooks. Pass `--cwd root` to exercise repository-root startup instead of section-folder startup. `--engine kernel` additionally exercises actual Jupyter kernels using local IPC, when the environment permits local sockets. Fresh-Python execution is not kernel, widget-rendering, or frontend verification. It does not execute the optional download, ClearML, gateway, or CUDA branches.

The unit tests cover deterministic data splits, shape/range/label handling, bounded download parsing and corruption checks, EMNIST mapping/orientation adapters, numeric checkpoint round trips and rejection, drawing preprocessing, bounded public-document retrieval/abstention, report schemas, chronological leakage safeguards, threshold selection, pandas/SQLite consistency, and both supported working directories.

### Verification scope

Local checks cover notebook schemas, Python syntax, repository-local links, helper tests, and fresh-process offline cell execution. At implementation time, all **28 section unit tests passed**, and all 13 follow-on notebooks executed in fresh Python processes from both the section folder and repository root. The course validator checked 31 notebook schemas and local links. Review plots included the held-out confusion matrix, nearest-neighbor gallery, and telemetry series.

Real MNIST and Fashion-MNIST downloads passed published compressed-file checksums, parsing, and bounded CPU training on 1,000 training / 200 test examples each. The real UCI Bike Sharing download also ran through the 731-day chronological modeling workflow. These small smoke runs do not establish benchmark performance. A bounded logistic run can emit a convergence warning at its iteration cap; it remains visible rather than suppressed.

Key test versions: Python 3.12, NumPy 2.3.5, pandas 2.2.3, scikit-learn 1.8.0, matplotlib 3.10.8, Pillow 12.3.0, nbformat 5.11.1, and ipywidgets 8.1.9. The supported ranges are not a lockfile or a promise of identical scores across environments. No live teaching tenant, Jupyter kernel/frontend, launch/restore, GPU, ClearML server, or live gateway is claimed verified. The validation environment blocks both TCP and IPC kernel sockets; the Python-process mode tests the same ordered code cells without those sockets. Real EMNIST data, optional PyTorch training, and interactive widget rendering remain unverified; the EMNIST adapter tests use mocked data and are not proof of remote availability. Preserve the introduction's deployment TODOs until someone performs those checks in the actual teaching environment.

**Back:** [Jupyter introduction](05_Running_Jupyter_Lab.ipynb) · **Next interface:** [SSH](../06_SSH/06_Running_an_SSH_Session.ipynb) · **Course:** [Orientation](../00_Orientation/00_Orientation.ipynb) · [Root README](../README.md)
