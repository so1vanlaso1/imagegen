#!/usr/bin/env bash
set -Eeuo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
source "$ROOT/config/versions.env"
trap 'echo "Setup failed at line $LINENO. Fix the reported error and rerun ./setup.sh; downloads resume." >&2' ERR

if [[ "${1:-}" == --help ]]; then
  echo 'Usage: ./setup.sh [--skip-models]'
  echo 'PYTHON_BIN=/path/to/python3.12 selects Python. ALLOW_LOW_RAM=1 permits <24 GiB RAM.'
  exit 0
fi
if [[ "$(uname -s)" != Linux ]]; then
  echo 'This installer targets Ubuntu/Debian Linux with an NVIDIA GPU (your Vast.ai PC).' >&2
  exit 1
fi
if [[ $# -gt 1 || ( $# -eq 1 && "$1" != --skip-models ) ]]; then
  echo 'Usage: ./setup.sh [--skip-models]' >&2; exit 1
fi

# Install ordinary host prerequisites; never replace the NVIDIA host driver.
if command -v apt-get >/dev/null; then
  elevated=()
  if [[ $EUID -ne 0 ]]; then
    command -v sudo >/dev/null || { echo 'Install git curl python3-venv libgl1 libglib2.0-0 first.' >&2; exit 1; }
    elevated=(sudo)
  fi
  "${elevated[@]}" apt-get update
  "${elevated[@]}" env DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    git curl ca-certificates python3 python3-venv libgl1 libglib2.0-0 tmux
fi
for command in git curl nvidia-smi; do
  command -v "$command" >/dev/null || { echo "Missing prerequisite: $command" >&2; exit 1; }
done
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv

if [[ -z "${PYTHON_BIN:-}" ]]; then
  for candidate in python3.12 python3.11 python3.10 python3; do
    if command -v "$candidate" >/dev/null && "$candidate" -c 'import sys; sys.exit(not ((3,10) <= sys.version_info[:2] <= (3,12)))'; then
      PYTHON_BIN="$candidate"; break
    fi
  done
fi
[[ -n "${PYTHON_BIN:-}" ]] || { echo 'Install Python 3.10, 3.11 or 3.12, then rerun setup.' >&2; exit 1; }
"$PYTHON_BIN" -c 'import sys; assert (3,10) <= sys.version_info[:2] <= (3,12), "Use Python 3.10–3.12"'
"$PYTHON_BIN" scripts/preflight.py
mkdir -p .runtime/bin .runtime/logs

checkout() {
  local url="$1" target="$2" commit="$3"
  if [[ ! -d "$target/.git" ]]; then
    [[ ! -e "$target" ]] || { echo "Expected git checkout at $target" >&2; return 1; }
    git init "$target"
    git -C "$target" remote add origin "$url"
  fi
  [[ -z "$(git -C "$target" status --porcelain --untracked-files=no)" ]] || {
    echo "Tracked changes in $target; save them before rerunning setup." >&2; return 1;
  }
  git -C "$target" fetch --depth 1 origin "$commit"
  git -C "$target" checkout --detach "$commit"
}
checkout https://github.com/Comfy-Org/ComfyUI.git .runtime/ComfyUI "$COMFYUI_COMMIT"
checkout https://github.com/leejet/ComfyUI-GGUF.git .runtime/ComfyUI/custom_nodes/ComfyUI-GGUF "$GGUF_COMMIT"
[[ -x .venv/bin/python ]] || "$PYTHON_BIN" -m venv .venv
PY="$ROOT/.venv/bin/python"
"$PY" -m pip install --upgrade pip
"$PY" -m pip install "torch==$TORCH_VERSION" "torchvision==$TORCHVISION_VERSION" --index-url "$TORCH_INDEX"
# Constraints keep upstream dependency resolution from replacing the CUDA build.
printf 'torch==%s\ntorchvision==%s\n' "$TORCH_VERSION" "$TORCHVISION_VERSION" > .runtime/constraints.txt
"$PY" -m pip install -c .runtime/constraints.txt -r .runtime/ComfyUI/requirements.txt \
  -r .runtime/ComfyUI/custom_nodes/ComfyUI-GGUF/requirements.txt -r requirements-tools.txt
"$PY" -m pip check
"$PY" -c 'import torch; assert torch.cuda.is_available(), "CUDA unavailable: check Vast GPU access / NVIDIA driver (CUDA 12.8 compatible driver required)"; print("CUDA ready:", torch.cuda.get_device_name(0)); print(torch.ones(1, device="cuda"))'

case "$(uname -m)" in
  x86_64) arch=amd64; digest="$CLOUDFLARED_AMD64_SHA256" ;;
  aarch64|arm64) arch=arm64; digest="$CLOUDFLARED_ARM64_SHA256" ;;
  *) echo 'Unsupported CPU architecture for cloudflared.' >&2; exit 1 ;;
esac
binary="$ROOT/.runtime/bin/cloudflared"
if [[ ! -f "$binary" ]] || ! echo "$digest  $binary" | sha256sum --check --status; then
  curl --fail --location --retry 5 \
    "https://github.com/cloudflare/cloudflared/releases/download/$CLOUDFLARED_VERSION/cloudflared-linux-$arch" -o "$binary.part"
  echo "$digest  $binary.part" | sha256sum --check
  mv "$binary.part" "$binary"
fi
chmod +x "$binary"

if [[ "${1:-}" != --skip-models ]]; then "$PY" scripts/download_models.py; fi
mkdir -p .runtime/user/default/workflows .runtime/ComfyUI/input
cp workflows/*.json .runtime/user/default/workflows/
"$PY" -m pip freeze > .runtime/installed-requirements.txt
echo 'Setup complete. Run ./run.sh. Workflows are available in the ComfyUI Workflows sidebar.'
