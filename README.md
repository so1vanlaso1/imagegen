# Qwen Image 2.1 Q8 on ComfyUI

Clone this project onto an NVIDIA Linux PC and run:

```bash
git clone <YOUR_REPO_URL> imagegen
cd imagegen
./setup.sh
./run.sh
```

`setup.sh` installs ComfyUI, the Qwen 2.1 compatible `leejet/ComfyUI-GGUF` fork, CUDA PyTorch, cloudflared, and these exact model files:

| Component | File | Download size |
| --- | --- | --- |
| Diffusion, **Q8_0** | `qwen-image-2.1-UC-Q8_0.gguf` | 7.59 GB |
| Text + vision encoder | `qwen3vl_8b_int8_convrot.safetensors` | 9.35 GB |
| VAE | `qwen_image_2.1_vae_bf16.safetensors` | 0.68 GB |

The diffusion model stays Q8; the separate encoder uses INT8 to reduce memory. Downloads resume on rerun and are verified against pinned SHA256 hashes. Model caches, environments, inputs, outputs and logs are ignored by Git. Weights are cached once and linked into ComfyUI's model folders.

## Vast.ai requirements

- RTX 3060 **12 GB VRAM**, with at least **32 GB system RAM recommended**. 12 GB VRAM and 12 GB system RAM are different constraints. The installer rejects less than 24 GiB of host/container RAM unless you deliberately set `ALLOW_LOW_RAM=1`; that override does not guarantee it will fit.
- Ubuntu 22.04/24.04 or a comparable Debian-based GPU image with Python **3.10–3.12**. Pick a CUDA 12.8 compatible NVIDIA driver. The PyTorch wheel includes its CUDA runtime; setup does not install or replace host drivers.
- Provision **60+ GB disk** for about 17.6 GB of model downloads, Python/CUDA dependencies, caches and generated images. More space is useful for outputs.
- Root or sudo for apt packages. Outbound HTTPS to GitHub, Hugging Face, PyTorch and Cloudflare. No inbound port mapping is required for the default tunnel.

Setup checks that CUDA can execute a tensor before downloading models. On a template with several Python installations, select one with `PYTHON_BIN=python3.12 ./setup.sh`. If that interpreter's venv module is missing, install the matching `python3.12-venv` package.

## Open ComfyUI from anywhere

`./run.sh` waits for ComfyUI and required nodes to load, then prints:

```text
PUBLIC COMFYUI URL (no login): https://...trycloudflare.com
```

Open that URL. It is also written to `.runtime/public-url.txt`. The default binds ComfyUI to localhost and exposes it through an anonymous Cloudflare Quick Tunnel. **Anyone with the URL can use your GPU, upload inputs, and access ComfyUI's served images/API.** Authentication is intentionally absent as requested. Stop with Ctrl+C; both processes stop and the URL file is removed. A new run gets a new hostname. Cloudflare Quick Tunnels have no uptime guarantee and are intended for temporary use.

Keep it running when your SSH connection closes:

```bash
tmux new -s imagegen
./run.sh
# Detach: Ctrl+B, then D. Return later with:
tmux attach -t imagegen
```

For a stable Vast.ai address, map container TCP port `8188` when renting the instance, then run `./run.sh --direct`. Open `http://<VAST_PUBLIC_IP>:<MAPPED_EXTERNAL_PORT>`. This also has no login and uses HTTP unless you add a TLS proxy. `./run.sh --local` disables public access. `--port 8190` changes the container/local port; update your mapping accordingly.

## Prompt with or without a reference

Setup installs both workflows in the **Workflows sidebar**. You can also drag these JSON files into the ComfyUI canvas:

- [Text to image](workflows/qwen21_q8_text_to_image.json): write a prompt in **Write your prompt / edit instruction here**, set width/height in **Output size**, then Run.
- [Reference image](workflows/qwen21_q8_reference_image.json): upload/select an image in **Upload your reference image**, write the edit instruction, then Run. Tell it what to preserve, such as subject identity, pose or product details, and what to change. This is native reference conditioning through both the vision encoder and VAE. It samples a new output rather than simply adding noise to your input.

Use the text workflow when you have no reference. The reference workflow requires an uploaded image; it is not a placeholder image that you have to remove. Its output follows the reference aspect ratio. `resolution=1024` means approximately one megapixel while preserving that ratio. Keep it at 1024 initially; 768 reduces memory use. Multiple references can be added through the native encoder's image inputs, but the provided 3060 workflow starts with one to limit memory.

Both workflows use **40 steps, Euler, simple, CFG 1, denoise 1, batch 1**, with randomized seeds in the UI. CFG 1 follows the upstream Qwen 2.1 path; negative prompts have no effect at CFG 1. Set the sampler seed control to fixed to reproduce a result.

For quality, describe the subject, composition, medium, lighting, colors and required details clearly. Put any desired visible text in quotation marks. For an edit, state the changes and what should remain consistent. No extra prompt-enhancer model or acceleration LoRA is installed.

Start at **1024 × 1024** for the 3060. After a successful run, try 1536 or 2048 for more detail, keeping dimensions divisible by 32. Native 2K is supported by the model, but Q8 at 2K on this GPU is **not guaranteed to fit**. More steps or resolution do not guarantee a better image. The first run includes model loading; CPU text/vision encoding and offload may take substantial time. No generation-time estimate has been measured on a 3060.

The encoder runs on CPU, lossless prefix caching uses system RAM, and diffusion uses classic low-VRAM offload. Tiled VAE decoding saves VRAM but may show subtle seams. To use a full decode once memory permits, replace **VAEDecodeTiled** with **VAEDecode** and reconnect the same inputs/output.

PNG images with workflow metadata are saved in `.runtime/ComfyUI/output/Qwen21/`. Back up outputs and `.runtime/user/` before destroying a rental. Reference uploads are in `.runtime/ComfyUI/input/`.

## Verify on the GPU

```bash
./run.sh --check
# With ./run.sh running in another terminal:
.venv/bin/python scripts/smoke_test.py
.venv/bin/python scripts/smoke_test.py --reference /path/to/reference.png
```

The smoke tests submit an actual 512-pixel, 4-step job, wait for execution, and fail on a node error or missing output. They test model execution, not image quality. For the full settings, run the supplied UI workflow. Pass `--url https://...trycloudflare.com` to the smoke test to exercise public access as well.

## Troubleshooting

- **CUDA out of memory:** lower width/height or reference resolution to 768, run one image at a time, and stop other GPU processes. If the failure is during VAE encode/decode, restart with `./run.sh --cpu-vae`. For heavier diffusion offload try `./run.sh --novram --cpu-vae`; it retains Q8 but runs slower. Low system RAM cannot be fixed merely by lowering VRAM use.
- **Killed / exit 137:** check the Vast container's RAM limit and host available RAM. Rent more RAM. Swap, where supported by the host, can be very slow and is not created automatically.
- **Missing node / unknown architecture:** rerun setup; this project pins the Qwen 2.1 compatible GGUF fork and ComfyUI revision together. Avoid replacing the fork with an older version.
- **No public URL:** inspect `.runtime/logs/tunnel.log`. The tunnel uses HTTP/2 over TCP for hosts that block UDP. Use `--direct` and a Vast port mapping if Cloudflare is unavailable.
- **Interrupted installation:** rerun `./setup.sh`; successful model files are hash checked and skipped. Use `./setup.sh --skip-models` to install only software, then `.venv/bin/python scripts/download_models.py` to fetch models later. `run.sh` rejects missing/incomplete files.
- **Logs:** `.runtime/logs/comfyui.log` and `.runtime/logs/tunnel.log`. `run.sh` rejects duplicate runs from this checkout and occupied ports. When either process exits, the other is stopped too.

## Versions and validation

Pins are in [config/versions.env](config/versions.env), with the model revision, sizes and hashes in [config/models.json](config/models.json). Upstream Python dependency ranges are resolved at setup time; `.runtime/installed-requirements.txt` records the resulting environment. This is not a complete lockfile for every transitive package.

Regenerate UI/API workflow files after changing the builder:

```bash
python3 scripts/build_workflows.py
python3 -m unittest discover -s tests -v
bash -n setup.sh run.sh
```

The development machine has no NVIDIA GPU. Both API workflows passed the pinned ComfyUI source's `validate_prompt` on CPU with placeholder model files and a test reference image (Mac PyTorch 2.14.1). Four offline tests cover graph wiring, checksum corruption, missing nodes and supervisor cleanup; shell syntax checks also passed. These checks do not load the real weights or verify the pinned Linux CUDA runtime. Actual Q8 inference, image quality and 3060 memory/performance must be checked with the commands above on the rental.

Sources: [model and companion files](https://huggingface.co/abenzerps/Qwen-Image-2.1-Uncensored-GGUF), [official model capabilities and sampling examples](https://huggingface.co/Qwen/Qwen-Image-2.1), [official ComfyUI text workflow](https://github.com/Comfy-Org/workflow_templates/blob/main/templates/image_qwen_image_2_1_t2i.json), [official edit workflow](https://github.com/Comfy-Org/workflow_templates/blob/main/templates/image_qwen_image_2_1_image_edit.json), [GGUF fork](https://github.com/leejet/ComfyUI-GGUF), [Cloudflare Quick Tunnels](https://developers.cloudflare.com/tunnel/get-started/quick-tunnels/). The model's upstream license is the Qwen Research License; weights are downloaded from the publisher rather than bundled here.
