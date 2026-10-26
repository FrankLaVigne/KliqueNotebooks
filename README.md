# KliqueNotebooks

A hands-on lab that teaches how to run [vLLM](https://docs.vllm.ai/) on [Klique](https://klique.ai/docs/latest/): deploy a model with the vLLM Model Deployment app, call its endpoint through the AI Application Gateway, and put it under governance with the LLM Gateway.

This is not a vLLM tutorial. Klique runs vLLM for you; the lab is about the Klique steps that turn a model name into a governed endpoint.

## Structure

| Section | Directory | Topic |
|---------|-----------|-------|
| 00 | `00_Orientation/` | Scope, mental model, and setup. Read this first. |
| 01 | `01_vLLM/` | Deploy a model with vLLM on Klique, generate a gateway token, and test the endpoint |
| 02 | `02_LLMGateway/` | Call the endpoint through the LLM Gateway under a profile, then call different models and Routed Resources |

## The Path

Every model you serve on Klique follows the same path. Each section of the lab covers one or more steps:

```
Model → vLLM Model Deployment → Model Endpoint → AI Application Gateway → LLM Gateway
 └──────────────── Section 01 ─────────────────────────────────────┘     └── Section 02 ──┘
```

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

   Section 2 needs no new secrets. The LLM Gateway URL is derived from `KLIQUE_BASE_URL` (it is always `<hostname>/llm`), and the profile and model names are set in the notebooks.

Each notebook installs its own requirements.

## Project Layout

```
KliqueNotebooks/
├── config.py              # Loads .env, exposes its values, and derives KLIQUE_LLM_GATEWAY_URL
├── .env                   # Endpoint and token (not tracked in git)
├── 00_Orientation/        # Section 00
├── 01_vLLM/               # Section 01
│   └── 01_Deploy_a_Model_with_vLLM.ipynb
└── 02_LLMGateway/         # Section 02
    ├── 02a_Use_the_LLM_Gateway.ipynb
    └── 02b_Models_and_Routed_Resources.ipynb   # Several models, one gateway; rule-based routing
```

## License

[MIT](LICENSE)
