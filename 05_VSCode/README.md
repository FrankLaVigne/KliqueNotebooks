# Hands-on VS Code labs

Start with [05: Running VS Code](05_Running_VS_Code.ipynb), then choose:

| Notebook | Companion code | Offline validation |
|---|---|---|
| [05b: Governed AI Playground](05b_Build_a_Governed_AI_Playground.ipynb) | `playground.py`, `gateway_client.py` | `test_gateway_playground.py` |
| [05c: Chat with This Repository](05c_Chat_with_This_Repository.ipynb) | `rag_lab.py` | `test_rag_evaluation.py` |
| [05d: Debug an Agent](05d_Debug_an_Agent.ipynb) | `agent_lab.py` | `test_agent_lab.py` |
| [05e: Run a Model Bake-Off](05e_Run_a_Model_Bake_Off.ipynb) | `evaluate_models.py`, `eval_questions.json` | `test_rag_evaluation.py` |
| [05f: Train, Stop, Restore, Continue](05f_Train_Stop_Restore_Continue.ipynb) | `train_classifier.py` | `test_training.py` |

These notebooks intentionally contain Markdown walkthroughs, matching the existing Part 05 and 06b conventions. Run their labelled terminal commands in the remote IDE. **Run All** is not a deployment or live-call shortcut. Each companion file is complete executable code, not a placeholder for the learner to assemble.

## Setup

Use Python 3.10+ in an approved environment. Run commands from the repository root. Reuse an appropriate venv or create one, then:

```bash
python -m pip install -r 05_VSCode/requirements-dev.txt
python -m pip check
```

For training, follow 05f's CPU/GPU-specific installation instructions and use the matching PyTorch build. `requirements-training.txt` documents the supported package ranges; it does not select a GPU wheel or install host drivers.

Live gateway configuration is documented in [`gateway.env.example`](gateway.env.example). Put real values only in the ignored root `.env` or approved injected environment variables. The helper does not import or change the older root `config.py`.

- `KLIQUE_BASE_URL` derives the HTTPS gateway host at `/llm`; explicit `KLIQUE_LLM_GATEWAY_URL` overrides it.
- `KLIQUE_GATEWAY_TOKEN` and `KLIQUE_PROFILE` supply the server-side identity and profile header.
- `KLIQUE_MODEL_ALIASES` is an explicit comma-separated allowlist from that profile. It is not the direct endpoint name.
- Optional rate/budget error codes must come from observed Section 03 responses. Unknown 429 responses stay ambiguous.

Existing environment variables override file values. Only simple `KEY=value` or quoted-value lines are accepted; no shell expansion is performed. No secret value is printed. Do not commit secrets or include them in screenshots. Ignored files may still be part of a workspace snapshot.

## Offline checks

```bash
python 05_VSCode/validate_labs.py
python -m unittest discover -s 05_VSCode -p 'test_gateway_playground.py' -v
python -m unittest discover -s 05_VSCode -p 'test_agent_lab.py' -v
python -m unittest discover -s 05_VSCode -p 'test_rag_evaluation.py' -v
# Run this in the approved training environment:
python -m unittest discover -s 05_VSCode -p 'test_training.py' -v
```

`validate_labs.py` checks notebook schema, local file links, and Python/bash example syntax without executing examples. The tests use fabricated responses, temporary directories, or the bundled digits dataset. They never call the Klique gateway. The Gradio construction test skips only if Gradio is absent; install the core requirements to cover it.

CLI smoke checks that do not require credentials:

```bash
python 05_VSCode/playground.py --check
python 05_VSCode/rag_lab.py 'How do I restore my VS Code workspace?'
python 05_VSCode/agent_lab.py
python 05_VSCode/evaluate_models.py --models alias-a alias-b
```

The playground uses an explicitly labelled fixture, RAG performs only local retrieval, the agent replays fabricated responses, and evaluation prints a request plan. **Real gateway calls require `--live`.** Live data and costs depend on your configured aliases and policies.

## Verification record

Locally verified when these labs were added:

- All **50 offline unit tests passed**: 11 gateway/playground, 20 agent, 12 RAG/evaluation, and 7 CPU training tests.
- All six Part 05 notebooks passed nbformat schema and local-link checks; companion files and fenced Python/bash examples parsed.
- Playground loopback HTTP and Gradio client invocation succeeded with the fixture, including an empty-input rejection and repeated invocation. The server was stopped afterward.
- Retrieval, evaluation dry-run, and agent/playground fixture commands succeeded. A mocked evaluation chart was generated and visually reviewed.
- CPU training completed epochs 1–3, restarted in another Python process, resumed at epoch 4, and completed epoch 8. Regression tests also compare uninterrupted versus split/resumed model, optimizer, RNG and metric state exactly under the fixed test environment.
- `pip check` passed for the core environment. Python static checks and `git diff --check` passed.

Key test versions were Python 3.12, Gradio 5.50.0, scikit-learn 1.9.1, nbformat 5.11.1, and CPU PyTorch 2.15.0+cpu. Supported package ranges are not a universal reproducibility guarantee: record the actual approved image and versions when you run a lab. Training explicitly rejects incompatible saved environments/configurations.

**Not verified against a live Klique account:** real gateway/model responses, AI Usage attribution, browser access through a particular VS Code deployment, Gradio Launcher deployment, cloud workspace stop/restore, and GPU execution. Those steps remain clearly labelled, with checkpoints and administrator prerequisites. Local tests do not establish those deployment-specific outcomes. No repository CI workflow was added or implied.

## Safety and cleanup

- Gradio defaults to remote loopback and `share=False`. An authenticated preview route must be confirmed; arbitrary VS Code forwarding is not assumed.
- The optional launcher uses one shared server-side gateway identity. Viewer authentication does not automatically give per-viewer model billing attribution.
- RAG reads only four allowlisted documentation files, Markdown cells only, without symlink traversal. Selected excerpts leave the workspace only with live generation. Citation checks do not prove factual support.
- Evaluation caps planned requests at 24, records failures/unknown usage, stops on governance/auth/transport failures, and never invents pricing or throughput metrics.
- The agent has only bounded local arithmetic, teaching-inventory lookup, and UTC clock tools. It does not execute model-generated Python or shell.
- Training loads only self-created, validated checkpoints using `weights_only=True`, with exclusive writers and atomic replacement. Files restore; processes do not.
- Generated results/checkpoints go under ignored `lab_outputs/` directories. Stop your playground, optional launcher, IDE, and any separately owned model/GPU instances as directed by each notebook. No recursive cleanup is required.
