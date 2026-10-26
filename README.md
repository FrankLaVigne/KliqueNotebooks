# Klique Notebooks

Jupyter notebooks for working with [vLLM](https://docs.vllm.ai/) model endpoints deployed on [Klique](https://klique.ai/docs/latest/).

## Notebooks

| Notebook | What it does |
|---|---|
| [`vllm_klique_test_clean.ipynb`](vllm_klique_test_clean.ipynb) | Sends test requests to a model served by Klique's **vLLM Model Deployment** app, using vLLM's OpenAI-compatible `/v1/completions` and `/v1/chat/completions` routes. It shows how to add a system prompt, split the `<think>` block that reasoning models such as Qwen3 return from the final answer, and turn thinking off with `chat_template_kwargs`. |

## Requirements

- A running vLLM Model Deployment instance with an **Active** endpoint ([docs](https://klique.ai/docs/latest/docs/webapp/applications/apps_model_deployment)).
- An AI Application Gateway token ([how to generate one](https://klique.ai/docs/latest/docs/webapp/settings/webapp_settings_profile#ai-application-gateway-tokens)).
- A machine that can reach the endpoint's network.
- Python 3 with Jupyter. The notebook installs `requests` and `python-dotenv` itself.

## Setup

Create a `.env` file next to the notebook:

```
KLIQUE_BASE_URL=https://<your-endpoint>/v1
KLIQUE_MODEL=<your Model Endpoint Name>
KLIQUE_GATEWAY_TOKEN=<your token>
```

`.env` is listed in `.gitignore`, so your token stays out of the repo. Don't paste a token into a notebook cell: it's saved in the `.ipynb` file. Clear the outputs before you commit, too, because they can show your endpoint URL.

Then open the notebook and run the cells in order.

## License

[MIT](LICENSE)
