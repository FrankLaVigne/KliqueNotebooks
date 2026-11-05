"""Send the 6B lab workflow to ComfyUI, wait for it, and save the image.

Run it in the session while ComfyUI is running:

    cd ~/workspace/KliqueNotebooks/06_SSH
    python3 run_comfy_sample.py                      # asks for a prompt
    python3 run_comfy_sample.py "a blue sailboat"    # uses this prompt

It reads workflow_api.json from the current folder if there is one (for example,
your own export); otherwise it uses the copy next to this script. The image and
receipt are saved in the current folder. Uses only the Python standard library.
"""
import json
import sys
import time
from pathlib import Path
from urllib.error import HTTPError
from urllib.parse import urlencode, quote
from urllib.request import Request, urlopen

BASE = "http://127.0.0.1:8188"
DEFAULT_PROMPT = "a red barn in a green meadow, rolling hills, afternoon sunlight, watercolor illustration"

workflow_path = Path("workflow_api.json")
if not workflow_path.exists():
    workflow_path = Path(__file__).resolve().parent / "workflow_api.json"
if not workflow_path.exists():
    raise SystemExit("No workflow_api.json here or next to this script.")
print("Workflow:", workflow_path.resolve())
workflow = json.loads(workflow_path.read_text(encoding="utf-8"))
if not isinstance(workflow, dict) or not workflow or not all(
    isinstance(node, dict) and "class_type" in node and "inputs" in node
    for node in workflow.values()
):
    raise SystemExit("Export API format, not the canvas save format.")

# Identify nodes by class rather than assuming the editor's numeric IDs.
samplers = [n for n in workflow.values() if n["class_type"] == "KSampler"]
savers = [n for n in workflow.values() if n["class_type"] == "SaveImage"]
if len(samplers) != 1 or len(savers) != 1:
    raise SystemExit("Use the single-sampler, single-SaveImage lab workflow.")

# The sampler's "positive" input links to the text node to replace, e.g. ["6", 0].
positive_id = samplers[0]["inputs"]["positive"][0]
positive = workflow.get(positive_id)
if positive is None or positive["class_type"] != "CLIPTextEncode":
    raise SystemExit("The sampler's positive input is not a CLIP text node.")

if len(sys.argv) > 1:
    text = " ".join(sys.argv[1:]).strip()
else:
    print("Image prompt (press Enter for the default):")
    print("  default:", DEFAULT_PROMPT)
    text = input("> ").strip()
text = text or DEFAULT_PROMPT
print("Prompt:", text, flush=True)

positive["inputs"]["text"] = text
samplers[0]["inputs"]["seed"] = 43
savers[0]["inputs"]["filename_prefix"] = "klique_api"

def fetch(path, payload=None):
    data = None if payload is None else json.dumps(payload).encode()
    headers = {} if data is None else {"Content-Type": "application/json"}
    request = Request(BASE + path, data=data, headers=headers)
    try:
        with urlopen(request, timeout=30) as response:
            return response.read()
    except HTTPError as error:
        raise RuntimeError(f"HTTP {error.code}: {error.read().decode(errors='replace')}") from error

def get_json(path, payload=None):
    return json.loads(fetch(path, payload))

# Submit once. Do not automatically retry POST after a network timeout.
receipt = get_json("/prompt", {"prompt": workflow})
if receipt.get("error") or receipt.get("node_errors"):
    raise SystemExit(json.dumps(receipt, indent=2))
prompt_id = receipt["prompt_id"]
print("Submitted:", prompt_id, flush=True)
Path("prompt-receipt.json").write_text(json.dumps(receipt, indent=2), encoding="utf-8")

deadline = time.monotonic() + 600
record = None
while time.monotonic() < deadline:
    record = get_json("/history/" + quote(prompt_id, safe="")).get(prompt_id)
    if record is not None:
        status = record.get("status", {})
        if status.get("status_str") == "error":
            raise SystemExit(json.dumps(status, indent=2))
        if status.get("completed"):
            break
    time.sleep(2)
else:
    raise SystemExit("Timed out waiting; job may still run. Inspect queue/server before resubmitting.")

images = [
    image
    for output in record.get("outputs", {}).values()
    for image in output.get("images", [])
    if image.get("type") == "output"
]
if not images:
    raise SystemExit("Finished without an output image; inspect history and SaveImage.")
for index, image in enumerate(images, 1):
    query = urlencode({key: image[key] for key in ("filename", "subfolder", "type")})
    png = fetch("/view?" + query)
    if not png.startswith(b"\x89PNG\r\n\x1a\n"):
        raise SystemExit("Server response is not a PNG.")
    # Fixed local naming avoids trusting server-supplied paths.
    destination = Path(f"api-{prompt_id}-{index}.png")
    with destination.open("xb") as stream:
        stream.write(png)
    print("Saved:", destination.resolve(), "bytes:", len(png))
