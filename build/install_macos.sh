#!/usr/bin/env bash
# ============================================================================
# Point2CAD macOS Installation Script
#
# Installs all dependencies on macOS (Intel or Apple Silicon) so that the
# Point2CAD pipeline can run natively without Docker.
#
# Usage:
#   chmod +x build/install_macos.sh
#   ./build/install_macos.sh
#
# After installation, activate the environment and run:
#   source .venv/bin/activate
#   python -m point2cad.main --path_in <your_file> --device cpu
# ============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
VENV_DIR="$PROJECT_DIR/.venv"
PYTHON_MIN_VERSION="3.9"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

info()  { echo -e "${GREEN}[INFO]${NC} $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC} $*"; }
error() { echo -e "${RED}[ERROR]${NC} $*"; }

# ---- Detect architecture ----
ARCH="$(uname -m)"
info "Detected architecture: $ARCH"

if [[ "$ARCH" == "arm64" ]]; then
    info "Apple Silicon detected — MPS (Metal) acceleration will be available"
elif [[ "$ARCH" == "x86_64" ]]; then
    info "Intel Mac detected — will use CPU backend"
else
    warn "Unknown architecture: $ARCH — proceeding with CPU backend"
fi

# ---- Check for Homebrew ----
if ! command -v brew &> /dev/null; then
    error "Homebrew is not installed. Install it first:"
    echo '  /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"'
    exit 1
fi
info "Homebrew found: $(brew --prefix)"

# ---- Install system dependencies ----
info "Installing system dependencies via Homebrew..."
brew install cmake gmp mpfr boost spatialindex 2>/dev/null || true

# ---- Check Python version ----
PYTHON_CMD=""
for cmd in python3.11 python3.10 python3.9 python3; do
    if command -v "$cmd" &> /dev/null; then
        version=$("$cmd" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
        major=$("$cmd" -c "import sys; print(sys.version_info.major)")
        minor=$("$cmd" -c "import sys; print(sys.version_info.minor)")
        if [[ "$major" -ge 3 ]] && [[ "$minor" -ge 9 ]]; then
            PYTHON_CMD="$cmd"
            info "Using Python $version ($cmd)"
            break
        fi
    fi
done

if [[ -z "$PYTHON_CMD" ]]; then
    error "Python >= $PYTHON_MIN_VERSION is required. Install with:"
    echo "  brew install python@3.11"
    exit 1
fi

# ---- Create virtual environment ----
if [[ -d "$VENV_DIR" ]]; then
    warn "Virtual environment already exists at $VENV_DIR"
    read -p "  Recreate it? [y/N] " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        rm -rf "$VENV_DIR"
        "$PYTHON_CMD" -m venv "$VENV_DIR"
        info "Virtual environment recreated"
    fi
else
    "$PYTHON_CMD" -m venv "$VENV_DIR"
    info "Virtual environment created at $VENV_DIR"
fi

source "$VENV_DIR/bin/activate"
pip install --upgrade pip wheel setuptools

# ---- Install PyTorch ----
info "Installing PyTorch..."
pip install torch torchvision torchaudio

# Verify MPS availability on Apple Silicon
if [[ "$ARCH" == "arm64" ]]; then
    MPS_OK=$(python -c "import torch; print(torch.backends.mps.is_available())" 2>/dev/null || echo "False")
    if [[ "$MPS_OK" == "True" ]]; then
        info "MPS (Metal) backend is available — GPU acceleration enabled"
    else
        warn "MPS backend not available on this system — will use CPU"
    fi
fi

# ---- Install core dependencies ----
info "Installing Python dependencies..."

# Core deps that work on all platforms
pip install numpy scipy trimesh open3d pyvista ezdxf geomdl rtree tqdm

# Optional: E57 and LAS support for FARO scanners
info "Installing FARO scanner format support..."
pip install laspy || warn "laspy installation failed — LAS format unavailable"
pip install lazrs || warn "lazrs installation failed — LAZ compression unavailable"
pip install pye57 || warn "pye57 installation failed — E57 format unavailable (try: brew install xerces-c)"

# Optional: improved self-intersection resolution
info "Installing manifold3d for mesh boolean operations..."
pip install manifold3d || warn "manifold3d installation failed — will use fallback mesh clipping"

# ---- Attempt PyMesh (optional) ----
info "Attempting PyMesh installation (optional — trimesh fallback is available)..."
PYMESH_OK=false
pip install pymesh2 2>/dev/null && PYMESH_OK=true || true

if [[ "$PYMESH_OK" == "true" ]]; then
    info "PyMesh installed successfully"
else
    warn "PyMesh is not available — using trimesh fallback for mesh operations"
    warn "This is fine for most use cases. Mesh clipping quality may be slightly reduced."
fi

# ---- Install project in development mode ----
cd "$PROJECT_DIR"
if [[ -f setup.py ]] || [[ -f pyproject.toml ]]; then
    pip install -e . 2>/dev/null || true
fi

# ---- Verify installation ----
info ""
info "============================================"
info "  Installation complete!"
info "============================================"
info ""
info "Activate the environment:"
info "  source $VENV_DIR/bin/activate"
info ""

# Print device info
python -c "
from point2cad.device import select_device
device = select_device()
print(f'  Default compute device: {device}')
" 2>/dev/null || true

# Print backend info
python -c "
from point2cad.mesh_ops import BACKEND
print(f'  Mesh backend: {BACKEND}')
" 2>/dev/null || true

info ""
info "Run the pipeline:"
info "  python -m point2cad.main --path_in <your_file.e57> --output_2d"
info ""
info "Convert a FARO scan first:"
info "  python -m point2cad.input_adapter scan.e57 --voxel_size 0.01"
info ""
