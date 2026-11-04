<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://klique.ai/assets/images/klique-logo-white.svg">
    <img alt="Klique" src="https://klique.ai/assets/images/klique-logo-black.svg" height="48">
  </picture>
</p>

<p align="center"><img src="images/banner.svg" alt="Klique: the control plane for enterprise AI. Route, govern, orchestrate, run anywhere." width="100%"></p>

# KliqueNotebooks

A hands-on lab that teaches how to run [vLLM](https://docs.vllm.ai/) on [Klique](https://klique.ai/docs/latest/): deploy a model with the vLLM Model Deployment app, call its endpoint through the AI Application Gateway, put it under governance with the LLM Gateway, cap its usage and cost, run a tool-using agent on the governed endpoint, and work on managed compute through VS Code, SSH, Claude Code, and Claude Desktop.

This is not a vLLM tutorial. Klique runs vLLM for you; Sections 01–04 turn a model name into a governed endpoint, and Sections 05–08 cover developer sessions on managed compute.

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="images/platform-dark.svg">
    <img src="images/platform-light.svg" alt="Klique as a distribution hub: AI requests and workloads arrive, Govern checks them and holds anything over quota, Route labels requests with the right model, Orchestrate loads workloads onto the right compute, and they ship to models or compute" width="100%">
  </picture>
</p>

## Structure

| Section | Directory | Topic |
|---------|-----------|-------|
| 00 | `00_Orientation/` | Scope, mental model, and setup. Read this first. |
| 01 | `01_vLLM/` | Deploy a model with vLLM on Klique, generate a gateway token, and test the endpoint |
| 02 | `02_LLMGateway/` | Call the endpoint through the LLM Gateway under a profile, call different models and Routed Resources, and route requests by meaning with semantic routing |
| 03 | `03_Governance/` | Read usage, price your vLLM Resource, set a request rate limit and a wallet budget, and write a client that handles each kind of rejection |
| 04 | `04_Agents/` | Build a small tool-using agent whose every step is metered, rate-limited, and budgeted by the LLM Gateway |
| 05 | [`05_VSCode/`](05_VSCode/05_Running_VS_Code.ipynb) | Remote IDE introduction, then five hands-on labs: governed playground, repository RAG, agent debugging, model evaluation, and resumable training |
| 06 | [`06_SSH/`](06_SSH/06_Running_an_SSH_Session.ipynb) | [SSH introduction](06_SSH/06_Running_an_SSH_Session.ipynb), then [manually install ComfyUI](06_SSH/06b_Install_ComfyUI_in_an_SSH_Session.ipynb) and generate images through its UI and API |
| 07 | [`07_ClaudeCode/`](07_ClaudeCode/07_Running_Claude_Code.ipynb) | Running Claude Code: use a managed coding agent, review changes, and inspect usage |
| 08 | [`08_ClaudeDesktop/`](08_ClaudeDesktop/08_Running_Claude_Desktop.ipynb) | Running Claude Desktop: open the hosted desktop app, try a small task, and clean up |

## The Path

Every model you serve on Klique follows the same path. Each section of the lab covers one or more steps:

```
Model → vLLM Model Deployment → Model Endpoint → AI Application Gateway → LLM Gateway
 └──────────────── Section 01 ─────────────────────────────────────┘     └── Section 02 ──┘
```

Sections 03 and 04 build on the governed endpoint: 03 caps what it can cost, and 04 runs an agent through it.

Sections 05–08 add developer sessions on managed compute. They can be taken independently of the endpoint labs; their introductory exercises do not require launching a model deployment. Start with [Running VS Code](05_VSCode/05_Running_VS_Code.ipynb), then explore SSH and the two Claude interfaces.

### Build Something in VS Code

All five follow-on notebooks live alongside the introduction in `05_VSCode/`. They are guided Markdown walkthroughs with runnable Python files, offline tests, prerequisites, expected checkpoints, troubleshooting and cleanup.

| Lab | What you build | Live access needed |
|---|---|---|
| [05b: Governed AI Playground](05_VSCode/05b_Build_a_Governed_AI_Playground.ipynb) | Gradio app with selectable gateway aliases, response metadata and optional Gradio Launcher deployment | Gateway for live responses; approved browser route for UI access |
| [05c: Chat with This Repository](05_VSCode/05c_Chat_with_This_Repository.ipynb) | Local TF-IDF retrieval over allowlisted documentation, bounded context and checked citations | Gateway only for optional answer generation |
| [05d: Debug an Agent](05_VSCode/05d_Debug_an_Agent.ipynb) | Part 04-style agent in a normal module, mocked tests and a VS Code breakpoint exercise | None for tests; gateway for an optional bounded live run |
| [05e: Run a Model Bake-Off](05_VSCode/05e_Run_a_Model_Bake_Off.ipynb) | Versioned question suite, bounded comparisons, JSONL/CSV results and quality/latency plot | Two permitted profile aliases for live comparison |
| [05f: Train, Stop, Restore, Continue](05_VSCode/05f_Train_Stop_Restore_Continue.ipynb) | Small image classifier with full checkpoints and explicit restart in a restored workspace | CPU works offline; approved compatible GPU is optional |

The live gateway labs use explicit `KLIQUE_PROFILE` and `KLIQUE_MODEL_ALIASES` settings in addition to the existing endpoint URL and gateway token. See [`gateway.env.example`](05_VSCode/gateway.env.example). Aliases are profile names, not necessarily the direct endpoint's `KLIQUE_MODEL`. Live model calls require `--live`; fixtures and local tests do not contact the gateway.

Install [`requirements.txt`](05_VSCode/requirements.txt) for 05b–05e and the appropriate approved PyTorch build plus [`requirements-training.txt`](05_VSCode/requirements-training.txt) for 05f. Generated results/checkpoints belong in the ignored `lab_outputs/` directory under the persistent workspace, not Git. A workspace snapshot can contain ignored secrets; use approved secret provisioning and review snapshot access.

See [Part 05 support files and verification](05_VSCode/README.md) for commands and the exact scope of local testing.

The second SSH notebook is an end-to-end [manual ComfyUI lab](06_SSH/06b_Install_ComfyUI_in_an_SSH_Session.ipynb): verify GPU compatibility, download a checked SD 1.5 checkpoint, start a loopback-only server, generate an image in the browser, then export and run the workflow through its API. It requires administrator-confirmed SSH forwarding and an appropriate GPU environment.

## Getting Started

### Prerequisites

For the endpoint labs (Sections 00–04):

- Python 3 and JupyterLab
- A Klique WebApp login
- A Klique queue with a GPU agent, and a project you can launch into
- A machine that can reach the model endpoint's network

For Sections 05–08, follow each notebook's app-specific prerequisites. They require permitted apps, projects, compute queues, and approved environments; the Claude exercises also require your organization's provider/account setup.

### Setup

1. Clone the repo:

   ```
   git clone https://github.com/FrankLaVigne/KliqueNotebooks.git
   ```

2. Work through the notebooks in order, starting with `00_Orientation/00_Orientation.ipynb`.

3. In Section 1.6 you create a `.env` file in the project root. The endpoint notebooks read it through `config.py`, so you set it once:

   ```
   KLIQUE_BASE_URL=https://<your endpoint>/v1
   KLIQUE_MODEL=<your Model Endpoint Name>
   KLIQUE_GATEWAY_TOKEN=<your AI Application Gateway token>
   ```

   Sections 2 to 4 need no new secrets. The LLM Gateway URL is derived from `KLIQUE_BASE_URL` (it is always `<hostname>/llm`), and the profile and model names are set in the notebooks.

The executable endpoint notebooks install their own requirements. Sections 05–08 are guided Markdown walkthroughs. Run commands only in the local or remote terminal identified by each step. Descriptive screenshot placeholders mark the screens to capture later.

## Project Layout

```
KliqueNotebooks/
├── config.py              # Loads .env, exposes its values, and derives KLIQUE_LLM_GATEWAY_URL
├── .env                   # Endpoint and token (not tracked in git)
├── 00_Orientation/        # Section 00
├── 01_vLLM/               # Section 01
│   └── 01_Deploy_a_Model_with_vLLM.ipynb
├── 02_LLMGateway/         # Section 02
│   ├── 02a_Use_the_LLM_Gateway.ipynb
│   ├── 02b_Models_and_Routed_Resources.ipynb   # Several models, one gateway; rule-based routing
│   └── 02c_Semantic_Routing.ipynb              # Route requests by what they mean
├── 03_Governance/         # Section 03
│   └── 03_Govern_Usage_Cost_and_Limits.ipynb   # Usage, pricing, rate limits, wallet budgets
├── 04_Agents/             # Section 04
│   └── 04_Agents_on_a_Governed_Endpoint.ipynb  # A tool-using agent under gateway governance
├── 05_VSCode/             # Section 05
│   ├── 05_Running_VS_Code.ipynb              # Remote editor and session lifecycle
│   ├── 05b_Build_a_Governed_AI_Playground.ipynb
│   ├── 05c_Chat_with_This_Repository.ipynb
│   ├── 05d_Debug_an_Agent.ipynb
│   ├── 05e_Run_a_Model_Bake_Off.ipynb
│   └── 05f_Train_Stop_Restore_Continue.ipynb  # Companion scripts and tests in same folder
├── 06_SSH/                # Section 06
│   ├── 06_Running_an_SSH_Session.ipynb       # Remote shell and connection lifecycle
│   └── 06b_Install_ComfyUI_in_an_SSH_Session.ipynb # Manual install, SD1.5, SSH tunnel, UI and API
├── 07_ClaudeCode/         # Section 07
│   └── 07_Running_Claude_Code.ipynb          # Managed coding agent and usage visibility
└── 08_ClaudeDesktop/      # Section 08
    └── 08_Running_Claude_Desktop.ipynb       # Browser-accessible desktop application
```

## License

[MIT](LICENSE)
