"""Send the 6B lab workflow to ComfyUI, wait for it, and save the image.

Run it in the session, from ~/workspace/comfyui-lab, while ComfyUI is running:

    python3 run_comfy_sample.py                      # asks for a prompt
    python3 run_comfy_sample.py "a blue sailboat"    # uses this prompt

It uses workflow_api.json from this folder if there is one (your export, or the
6B.4 setup script's copy). Otherwise it uses the built-in 6B.7 lab workflow.
Uses only the Python standard library.
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

CKPT_NAME = "v1-5-pruned-emaonly-fp16.safetensors"

# The 6B.7 lab workflow in API format: node ID -> class_type and inputs.
LAB_WORKFLOW = {
    "3": {"class_type": "KSampler", "inputs": {
        "seed": 42, "steps": 20, "cfg": 7.0, "sampler_name": "euler",
        "scheduler": "normal", "denoise": 1.0,
        "model": ["4", 0], "positive": ["6", 0], "negative": ["7", 0], "latent_image": ["5", 0]}},
    "4": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": CKPT_NAME}},
    "5": {"class_type": "EmptyLatentImage", "inputs": {"width": 512, "height": 512, "batch_size": 1}},
    "6": {"class_type": "CLIPTextEncode", "inputs": {"text": DEFAULT_PROMPT, "clip": ["4", 1]}},
    "7": {"class_type": "CLIPTextEncode", "inputs": {"text": "text, watermark, blurry", "clip": ["4", 1]}},
    "8": {"class_type": "VAEDecode", "inputs": {"samples": ["3", 0], "vae": ["4", 2]}},
    "9": {"class_type": "SaveImage", "inputs": {"filename_prefix": "klique_api", "images": ["8", 0]}},
}

workflow_path = Path("workflow_api.json")
if workflow_path.exists():
    print("Workflow:", workflow_path.resolve())
    workflow = json.loads(workflow_path.read_text(encoding="utf-8"))
else:
    print("Workflow: built-in 6B.7 lab workflow (no workflow_api.json here)")
    workflow = LAB_WORKFLOW
    ckpt = Path("ComfyUI/models/checkpoints") / CKPT_NAME
    if Path("ComfyUI").is_dir() and not ckpt.exists():
        raise SystemExit(f"Missing {ckpt}. Download and verify it in 6B.4 first.")
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
