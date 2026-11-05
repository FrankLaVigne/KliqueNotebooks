"""Section 6B setup: download and verify the SD 1.5 checkpoint, then write the lab workflow.

Run it from ~/workspace/comfyui-lab, the folder that holds ComfyUI/:

    cd ~/workspace/comfyui-lab
    python3 ~/workspace/KliqueNotebooks/06_SSH/setup_comfyui_lab.py

It writes two copies of the 6B.7 workflow:
  ComfyUI/user/default/workflows/klique_lab_sd15.json  canvas format, listed under Workflows in ComfyUI
  workflow_api.json                                    API format, to compare with your 6B.8 export

The API copy matches 06_SSH/workflow_api.json in the repository, which 6B.9 sends.

Safe to rerun: a verified checkpoint is not downloaded again, and both workflow files are rewritten.
Uses only the Python standard library.
"""
import hashlib
import json
import sys
from pathlib import Path
from urllib.request import urlopen

COMFY = Path("ComfyUI")
CKPT_NAME = "v1-5-pruned-emaonly-fp16.safetensors"
CKPT_URL = "https://huggingface.co/Comfy-Org/stable-diffusion-1.5/resolve/main/" + CKPT_NAME
CKPT_SHA256 = "e9476a13728cd75d8279f6ec8bad753a66a1957ca375a1464dc63b37db6e3916"

# The lab's sample settings (6B.7)
POSITIVE = "a small red cabin beside a calm lake, pine trees, morning sunlight, watercolor illustration"
NEGATIVE = "text, watermark, blurry"
WIDTH, HEIGHT, BATCH = 512, 512, 1
SEED, STEPS, CFG, SAMPLER, SCHEDULER, DENOISE = 42, 20, 7.0, "euler", "normal", 1.0
PREFIX = "klique_ui"

if not COMFY.is_dir():
    sys.exit(f"No {COMFY}/ here. cd ~/workspace/comfyui-lab first.")


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


# 1. Checkpoint
ckpt = COMFY / "models" / "checkpoints" / CKPT_NAME
if "--no-download" in sys.argv:
    print("Skipping download (--no-download)")
elif ckpt.exists():
    if sha256(ckpt) != CKPT_SHA256:
        sys.exit(f"{ckpt} exists but its checksum is wrong. Inspect or delete it, then rerun.")
    print(f"✅ Checkpoint already present and verified: {ckpt}")
else:
    part = ckpt.with_name(CKPT_NAME + ".part")
    print(f"Downloading {CKPT_NAME} (about 2.1 GB)...")
    digest, done, next_report = hashlib.sha256(), 0, 0
    with urlopen(CKPT_URL, timeout=60) as response, part.open("wb") as out:
        total = int(response.headers.get("Content-Length") or 0)
        for block in iter(lambda: response.read(1024 * 1024), b""):
            out.write(block)
            digest.update(block)
            done += len(block)
            if done >= next_report:
                print(f"  {done / 2**30:.2f} / {total / 2**30:.2f} GiB", flush=True)
                next_report += 256 * 2**20
    if digest.hexdigest() != CKPT_SHA256:
        part.unlink()
        sys.exit(f"Checksum mismatch ({digest.hexdigest()}). Deleted the download; do not load it.")
    part.rename(ckpt)
    print(f"✅ Verified: {ckpt} ({ckpt.stat().st_size:,} bytes)")

# 2. API-format workflow (6B.8 / 6B.9): node ID -> class_type and inputs
api = {
    "3": {"class_type": "KSampler", "inputs": {
        "seed": SEED, "steps": STEPS, "cfg": CFG, "sampler_name": SAMPLER,
        "scheduler": SCHEDULER, "denoise": DENOISE,
        "model": ["4", 0], "positive": ["6", 0], "negative": ["7", 0], "latent_image": ["5", 0]}},
    "4": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": CKPT_NAME}},
    "5": {"class_type": "EmptyLatentImage", "inputs": {"width": WIDTH, "height": HEIGHT, "batch_size": BATCH}},
    "6": {"class_type": "CLIPTextEncode", "inputs": {"text": POSITIVE, "clip": ["4", 1]}},
    "7": {"class_type": "CLIPTextEncode", "inputs": {"text": NEGATIVE, "clip": ["4", 1]}},
    "8": {"class_type": "VAEDecode", "inputs": {"samples": ["3", 0], "vae": ["4", 2]}},
    "9": {"class_type": "SaveImage", "inputs": {"filename_prefix": PREFIX, "images": ["8", 0]}},
}

# 3. Canvas-format workflow (what the editor saves), same graph with layout
# links: [link id, from node, from slot, to node, to slot, type]
links = [
    [1, 4, 0, 3, 0, "MODEL"],
    [2, 5, 0, 3, 3, "LATENT"],
    [3, 4, 1, 6, 0, "CLIP"],
    [4, 6, 0, 3, 1, "CONDITIONING"],
    [5, 4, 1, 7, 0, "CLIP"],
    [6, 7, 0, 3, 2, "CONDITIONING"],
    [7, 3, 0, 8, 0, "LATENT"],
    [8, 4, 2, 8, 1, "VAE"],
    [9, 8, 0, 9, 0, "IMAGE"],
]


def node(node_id, node_type, pos, size, order, inputs=(), outputs=(), widgets=()):
    return {
        "id": node_id, "type": node_type, "pos": pos, "size": size, "flags": {},
        "order": order, "mode": 0,
        "inputs": [{"name": n, "type": t, "link": l} for n, t, l in inputs],
        "outputs": [{"name": n, "type": t, "links": ls, "slot_index": i}
                    for i, (n, t, ls) in enumerate(outputs)],
        "properties": {"Node name for S&R": node_type},
        "widgets_values": list(widgets),
    }


canvas = {
    "last_node_id": 9, "last_link_id": 9,
    "nodes": [
        node(4, "CheckpointLoaderSimple", [26, 474], [315, 98], 0,
             outputs=[("MODEL", "MODEL", [1]), ("CLIP", "CLIP", [3, 5]), ("VAE", "VAE", [8])],
             widgets=[CKPT_NAME]),
        node(5, "EmptyLatentImage", [473, 609], [315, 106], 1,
             outputs=[("LATENT", "LATENT", [2])], widgets=[WIDTH, HEIGHT, BATCH]),
        node(6, "CLIPTextEncode", [415, 186], [422, 164], 2,
             inputs=[("clip", "CLIP", 3)], outputs=[("CONDITIONING", "CONDITIONING", [4])],
             widgets=[POSITIVE]),
        node(7, "CLIPTextEncode", [413, 389], [425, 180], 3,
             inputs=[("clip", "CLIP", 5)], outputs=[("CONDITIONING", "CONDITIONING", [6])],
             widgets=[NEGATIVE]),
        node(3, "KSampler", [863, 186], [315, 262], 4,
             inputs=[("model", "MODEL", 1), ("positive", "CONDITIONING", 4),
                     ("negative", "CONDITIONING", 6), ("latent_image", "LATENT", 2)],
             outputs=[("LATENT", "LATENT", [7])],
             widgets=[SEED, "fixed", STEPS, CFG, SAMPLER, SCHEDULER, DENOISE]),
        node(8, "VAEDecode", [1209, 188], [210, 46], 5,
             inputs=[("samples", "LATENT", 7), ("vae", "VAE", 8)], outputs=[("IMAGE", "IMAGE", [9])]),
        node(9, "SaveImage", [1451, 189], [210, 270], 6,
             inputs=[("images", "IMAGE", 9)], widgets=[PREFIX]),
    ],
    "links": links, "groups": [], "config": {}, "extra": {}, "version": 0.4,
}

workflows = COMFY / "user" / "default" / "workflows"
workflows.mkdir(parents=True, exist_ok=True)
canvas_path = workflows / "klique_lab_sd15.json"
api_path = Path("workflow_api.json")
canvas_path.write_text(json.dumps(canvas, indent=2), encoding="utf-8")
api_path.write_text(json.dumps(api, indent=2), encoding="utf-8")
print(f"✅ Canvas workflow: {canvas_path}  (ComfyUI sidebar: Workflows > klique_lab_sd15)")
print(f"✅ API workflow:    {api_path.resolve()}  (compare in 6B.8)")
