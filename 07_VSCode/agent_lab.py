"""Part 04's tiny tool agent, extracted for offline debugging (Python 3.10+).

No network or configuration is loaded on import. Only --live creates a gateway
client. Reported token limits stop subsequent work; they are not prepaid budgets.
"""
from __future__ import annotations

import argparse
import ast
from copy import deepcopy
from datetime import datetime, timezone
import json
import math
import operator

INVENTORY = {
    "lab-a100": {"gpus": 8, "gpu_type": "A100 80GB", "hourly_rate_usd_per_gpu": 2.5},
    "lab-l40s": {"gpus": 4, "gpu_type": "L40S", "hourly_rate_usd_per_gpu": 1.2},
}
TASK = "Look up lab-a100, then calculate the cost of all its GPUs for 3 hours."
OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
       ast.Div: operator.truediv, ast.Pow: operator.pow, ast.Mod: operator.mod,
       ast.USub: operator.neg, ast.UAdd: operator.pos}


def calculator(expression):
    """Small bounded arithmetic language; never eval arbitrary Python."""
    if not isinstance(expression, str) or len(expression) > 200:
        raise ValueError("expression must be a string of at most 200 characters")
    tree = ast.parse(expression, mode="eval")
    if sum(1 for _ in ast.walk(tree)) > 60:
        raise ValueError("expression too complex")

    def evaluate(node):
        if isinstance(node, ast.Expression):
            value = evaluate(node.body)
        elif isinstance(node, ast.Constant) and type(node.value) in (int, float):
            value = node.value
        elif isinstance(node, ast.BinOp) and type(node.op) in OPS:
            left, right = evaluate(node.left), evaluate(node.right)
            if isinstance(node.op, ast.Pow) and abs(right) > 10:
                raise ValueError("exponent too large")
            value = OPS[type(node.op)](left, right)
        elif isinstance(node, ast.UnaryOp) and type(node.op) in OPS:
            value = OPS[type(node.op)](evaluate(node.operand))
        else:
            raise ValueError("only arithmetic is allowed")
        if type(value) not in (int, float) or not math.isfinite(value) or abs(value) > 1e12:
            raise ValueError("result outside the lab's numeric range")
        return value

    return evaluate(tree)


def lookup(key):
    return deepcopy(INVENTORY.get(key, {"error": "unknown inventory key"}))


def clock():
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")


TOOL_FUNCTIONS = {"calculator": calculator, "lookup": lookup, "clock": clock}
TOOLS = [
    {"type": "function", "function": {"name": name, "description": description,
     "parameters": {"type": "object", "properties": properties,
                    "required": list(properties), "additionalProperties": False}}}
    for name, description, properties in [
        ("calculator", "Calculate arithmetic, e.g. 8 * 2.5 * 3.", {"expression": {"type": "string"}}),
        ("lookup", "Look up teaching inventory: lab-a100 or lab-l40s.", {"key": {"type": "string"}}),
        ("clock", "Get current UTC time.", {}),
    ]
]
JSON_SYSTEM = (
    "Use these read-only tools: " + json.dumps(TOOLS) +
    '\nReply with exactly one JSON object: {"tool":"name","arguments":{...}} '
    'or {"final":"answer"}. Use one tool per reply.'
)


def run_tool(name, arguments):
    if not isinstance(name, str) or name not in TOOL_FUNCTIONS:
        return {"error": "unknown tool"}
    if not isinstance(arguments, dict):
        return {"error": "arguments must be a JSON object"}
    try:
        return {"result": TOOL_FUNCTIONS[name](**arguments)}
    except (TypeError, ValueError, SyntaxError, ArithmeticError, RecursionError) as error:
        return {"error": f"{type(error).__name__}: {error}"}


def strip_think(text):
    text = text or ""
    while "<think>" in text:
        start = text.index("<think>")
        end = text.find("</think>", start)
        text = text[:start] + ("" if end == -1 else text[end + 8:])
    return text.replace("</think>", "").strip()


def run_agent(client, task=TASK, *, model="mock", mode="json", max_steps=6,
              max_run_tokens=6000, max_parse_errors=2, max_tokens=256):
    """Inject a client with complete(messages, **kwargs); never retry a request.

    A client exception with status_code/kind is recorded, without its body or
    credentials. Unknown 429 remains ambiguous; do not infer wallet exhaustion.
    Missing usage fails closed, since the run cannot account for those tokens.
    """
    if mode not in ("json", "native"):
        raise ValueError("choose json or native explicitly")
    for value in (max_steps, max_run_tokens, max_parse_errors, max_tokens):
        if type(value) is not int or value < 1:
            raise ValueError("limits must be positive integers")
    history = [{"role": "user", "content": task}]
    if mode == "json":
        history.insert(0, {"role": "system", "content": JSON_SYSTEM})
    trace, tokens, parse_errors = [], {"prompt": 0, "completion": 0}, 0
    status, answer = "max_steps", None
    for step in range(1, max_steps + 1):
        row = {"step": step, "requests": 1, "tool": "", "note": "",
               "prompt_tokens": 0, "completion_tokens": 0}
        trace.append(row)
        try:
            reply = client.complete(deepcopy(history), model=model, max_tokens=max_tokens,
                                    temperature=0, tools=TOOLS if mode == "native" else None,
                                    tool_choice="auto" if mode == "native" else None)
        except Exception as error:
            # Unexpected programming errors still propagate to the debugger.
            if not hasattr(error, "status_code") or not hasattr(error, "kind"):
                raise
            row["note"] = f"gateway stop: HTTP {error.status_code}; kind={error.kind}"
            status = "stopped_by_gateway"
            break
        if not isinstance(reply, dict):
            status, row["note"] = "invalid_response", "Response is not an object"
            break
        usage = reply.get("usage") or {}
        if not isinstance(usage, dict):
            usage = {}
        if any(type(usage.get(key)) is not int or usage[key] < 0
               for key in ("prompt_tokens", "completion_tokens")):
            status, row["note"] = "usage_unavailable", "Cannot enforce observed-token accounting"
            break
        for key in tokens:
            count = usage[key + "_tokens"]
            tokens[key] += count  # DEBUG: inspect cumulative usage here.
            row[key + "_tokens"] = count
        try:
            message = reply["choices"][0]["message"]
            if not isinstance(message, dict):
                raise ValueError("message is not an object")
        except (KeyError, IndexError, TypeError, ValueError):
            status, row["note"] = "invalid_response", "Missing assistant message"
            break
        history.append(deepcopy(message) | {"role": "assistant"})
        # This is a post-response stop, not a guaranteed ceiling on billed tokens.
        if sum(tokens.values()) >= max_run_tokens:
            status, row["note"] = "token_limit", "Observed token threshold reached; no tool or next call"
            break
        try:
            if mode == "native":
                calls = message.get("tool_calls") or []
                if not isinstance(calls, list) or len(calls) > 4:
                    raise ValueError("at most four native calls per response")
                if not calls:
                    answer = strip_think(message.get("content"))
                    if not answer:
                        raise ValueError("empty final answer")
                    status, row["note"] = "done", "final answer"
                    break
                for call in calls:
                    function = call["function"]
                    if not isinstance(call["id"], str) or not call["id"]:
                        raise ValueError("missing tool_call_id")
                    arguments = json.loads(function["arguments"])
                    result = run_tool(function["name"], arguments)
                    row["tool"] += function["name"] + " "
                    history.append({"role": "tool", "tool_call_id": call["id"],
                                    "content": json.dumps(result)})
            else:
                obj = json.loads(strip_think(message.get("content")))
                if not isinstance(obj, dict):
                    raise ValueError("expected an object")
                if set(obj) == {"final"} and isinstance(obj["final"], str):
                    if not obj["final"].strip():
                        raise ValueError("empty final answer")
                    answer, status, row["note"] = obj["final"], "done", "final answer"
                    break
                if set(obj) != {"tool", "arguments"} or not isinstance(obj["tool"], str):
                    raise ValueError("expected tool and arguments, or final")
                result = run_tool(obj["tool"], obj["arguments"])
                row["tool"] = obj["tool"]
                history.append({"role": "user", "content": "TOOL RESULT: " + json.dumps(result)})
            row["note"] = json.dumps(result)
            parse_errors = 0
        except (ValueError, KeyError, TypeError, AttributeError):
            parse_errors += 1
            row["note"] = "Malformed tool response"
            if mode == "native" or parse_errors >= max_parse_errors:
                status = "parse_error"
                break
            history.append({"role": "user", "content": "Reply with exactly one valid JSON object as instructed."})
    return {"status": status, "answer": answer, "tokens": tokens, "trace": trace, "history": history}


class MockClient:
    """Replay responses/exceptions; capture calls without any network access."""
    def __init__(self, responses):
        self.responses, self.calls = iter(responses), []

    def complete(self, messages, **kwargs):
        self.calls.append({"messages": deepcopy(messages), **kwargs})
        response = next(self.responses)
        if isinstance(response, Exception):
            raise response
        return deepcopy(response)


def mock_reply(content, prompt=10, completion=5, tool_calls=None):
    message = {"role": "assistant", "content": content}
    if tool_calls is not None:
        message["tool_calls"] = tool_calls
    return {"choices": [{"message": message}],
            "usage": {"prompt_tokens": prompt, "completion_tokens": completion}}


def demo_client():
    return MockClient([
        mock_reply('{"tool":"lookup","arguments":{"key":"lab-a100"}}'),
        mock_reply('{"tool":"calculator","arguments":{"expression":"8 * 2.5 * 3"}}'),
        mock_reply('{"final":"$60.00 for 3 hours"}'),
    ])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", action="store_true", help="opt in to metered gateway requests")
    parser.add_argument("--mode", choices=["json", "native"], default="json")
    args = parser.parse_args()
    if args.live:
        from gateway_client import GatewayClient, GatewayConfig
        config = GatewayConfig.from_environment()
        client, model = GatewayClient(config), config.models[0]
    else:
        if args.mode != "json":
            parser.error("offline CLI demo uses JSON; native mode is covered by tests")
        client, model = demo_client(), "mock"
    run = run_agent(client, model=model, mode=args.mode)
    for row in run["trace"]:
        print(f"step {row['step']}: {row['tool'] or '-'} {row['note']}")
    print(f"status: {run['status']}; tokens: {sum(run['tokens'].values())}; answer: {run['answer']}")
    return 0 if run["status"] == "done" else 1


if __name__ == "__main__":
    raise SystemExit(main())
