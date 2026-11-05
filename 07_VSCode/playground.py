"""A single-turn playground. Defaults to a labelled offline fixture, never a model call."""
from __future__ import annotations

import argparse
import json
import time
from gateway_client import GatewayClient, GatewayConfig, GatewayError


class FixtureClient:
    """Shows the UI without credentials. This is not generated model output."""
    def complete(self, messages, *, model=None, max_tokens=256, temperature=0):
        return {"model": "offline-fixture", "choices": [{"message": {"content":
                "OFFLINE FIXTURE: the playground is wired correctly. Start with --live to call your gateway."},
                "finish_reason": "stop"}]}


def answer(client, aliases, prompt, alias, temperature, max_tokens):
    if not isinstance(prompt, str) or not prompt.strip():
        return "Enter a question.", {"sent": False}
    if len(prompt) > 4000:
        return "Keep this lab prompt under 4,000 characters.", {"sent": False}
    if alias not in aliases:
        return "Choose a configured model alias.", {"sent": False}
    started = time.perf_counter()
    try:
        response = client.complete([{"role": "system", "content": "Answer the user's question clearly and briefly."},
                                    {"role": "user", "content": prompt}], model=alias,
                                   temperature=float(temperature), max_tokens=int(max_tokens))
    except GatewayError as exc:
        return str(exc), {"requested_alias": alias, "error_kind": exc.kind,
                          "http_status": exc.status_code, "client_elapsed_seconds": round(time.perf_counter() - started, 3)}
    choice = response["choices"][0]
    content = choice["message"].get("content")
    if not isinstance(content, str) or not content.strip():
        content = "No text answer returned. Check finish_reason and model compatibility."
    return content, {"requested_alias": alias, "returned_model": response.get("model"),
                     "client_elapsed_seconds": round(time.perf_counter() - started, 3),
                     "usage_reported": response.get("usage"), "finish_reason": choice.get("finish_reason"),
                     "offline_fixture": isinstance(client, FixtureClient)}


def build_demo(client, aliases):
    import gradio as gr
    with gr.Blocks(title="Klique: Governed AI Playground", analytics_enabled=False) as demo:
        gr.Markdown("# Governed AI Playground\nSingle-turn lab. Use public sample prompts only. "
                    "Each live click sends one request using this app's shared gateway identity.")
        if isinstance(client, FixtureClient):
            gr.Markdown("**OFFLINE FIXTURE MODE: no model calls and no token usage.**")
        alias = gr.Dropdown(list(aliases), value=aliases[0], label="Gateway profile alias")
        prompt = gr.Textbox(label="Prompt", lines=3, value="Explain why a saved workspace is not a running process.")
        with gr.Row():
            temperature = gr.Slider(0, 1, value=0, step=0.1, label="Temperature")
            max_tokens = gr.Slider(32, 512, value=256, step=32, label="Maximum output tokens")
        submit = gr.Button("Send one request", variant="primary")
        output = gr.Textbox(label="Answer", lines=7)
        metadata = gr.JSON(label="Observed request metadata (null means not reported)")
        submit.click(lambda p, a, t, m: answer(client, aliases, p, a, t, m),
                     [prompt, alias, temperature, max_tokens], [output, metadata], concurrency_limit=1)
    return demo


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", action="store_true", help="Send real model requests when the button is clicked")
    parser.add_argument("--check", action="store_true", help="Run one public sample request without starting a UI")
    parser.add_argument("--host", choices=["127.0.0.1", "0.0.0.0"], default="127.0.0.1",
                        help="All-interface binding is ONLY for an administrator-approved authenticated launcher")
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()
    if not 1024 <= args.port <= 65535:
        parser.error("Choose a port from 1024 to 65535")
    if args.live:
        config = GatewayConfig.from_environment()
        client, aliases = GatewayClient(config), config.models
    else:
        client, aliases = FixtureClient(), ("offline-fixture",)
    if args.check:
        content, metadata = answer(client, aliases, "What survives when a development session stops?", aliases[0], 0, 128)
        print(content)
        print(json.dumps(metadata, indent=2))
        if metadata.get("error_kind"):
            raise SystemExit(1)
        return
    # No public share tunnel, file paths or debug tracebacks. Access policy is enforced outside this app.
    build_demo(client, aliases).queue(max_size=8).launch(server_name=args.host, server_port=args.port,
                                                      share=False, show_error=False, allowed_paths=[],
                                                      max_file_size="1mb")


if __name__ == "__main__":
    main()
