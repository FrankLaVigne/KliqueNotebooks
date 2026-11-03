<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://klique.ai/assets/images/klique-logo-white.svg">
    <img alt="Klique" src="https://klique.ai/assets/images/klique-logo-black.svg" height="48">
  </picture>
</p>

<p align="center"><img src="images/banner.svg" alt="Klique: the control plane for enterprise AI. Route, govern, orchestrate, run anywhere." width="100%"></p>

# KliqueNotebooks

A hands-on lab that teaches how to run [vLLM](https://docs.vllm.ai/) on [Klique](https://klique.ai/docs/latest/): deploy a model with the vLLM Model Deployment app, call its endpoint through the AI Application Gateway, put it under governance with the LLM Gateway, cap its usage and cost, and run a tool-using agent on the governed endpoint.

This is not a vLLM tutorial. Klique runs vLLM for you; the lab is about the Klique steps that turn a model name into a governed endpoint.

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

## The Path

Every model you serve on Klique follows the same path. Each section of the lab covers one or more steps:

```
Model → vLLM Model Deployment → Model Endpoint → AI Application Gateway → LLM Gateway
 └──────────────── Section 01 ─────────────────────────────────────┘     └── Section 02 ──┘
```

Sections 03 and 04 build on the governed endpoint: 03 caps what it can cost, and 04 runs an agent through it.

## Getting Started

### Prerequisites

- Python 3 and JupyterLab
- A Klique WebApp login
- A Klique queue with a GPU agent, and a project you can launch into
- A machine that can reach the model endpoint's network

### Setup

1. Clone the repo:

   ```
   git clone https://github.com/FrankLaVigne/KliqueNotebooks.git
   ```

2. Work through the notebooks in order, starting with `00_Orientation/00_Orientation.ipynb`.

3. In Section 1.6 you create a `.env` file in the project root. Every notebook reads it through `config.py`, so you set it once:

   ```
   KLIQUE_BASE_URL=https://<your endpoint>/v1
   KLIQUE_MODEL=<your Model Endpoint Name>
   KLIQUE_GATEWAY_TOKEN=<your AI Application Gateway token>
   ```

   Sections 2 to 4 need no new secrets. The LLM Gateway URL is derived from `KLIQUE_BASE_URL` (it is always `<hostname>/llm`), and the profile and model names are set in the notebooks.

Each notebook installs its own requirements.

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
└── 04_Agents/             # Section 04
    └── 04_Agents_on_a_Governed_Endpoint.ipynb  # A tool-using agent under gateway governance
```

## License

[MIT](LICENSE)
