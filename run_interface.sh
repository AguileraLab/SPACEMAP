#!/bin/bash
# run_interface.sh: Start Streamlit GUI with the correct environment and libraries.

# Get the directory of this script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

# Activate the 'spacemap' Conda environment safely if available
activate_spacemap_env() {
    local ENV_NAME="spacemap"

    # Check if target environment is already active
    if [ "${CONDA_DEFAULT_ENV:-}" = "$ENV_NAME" ] || [[ "${CONDA_PREFIX:-}" == *"/$ENV_NAME" ]]; then
        echo "Conda environment '$ENV_NAME' is already active."
        return 0
    fi

    # Attempt to initialize Conda shell hook for non-interactive bash sessions
    if command -v conda >/dev/null 2>&1; then
        eval "$(conda shell.bash hook 2>/dev/null)" || true
    elif [ -n "${CONDA_EXE:-}" ]; then
        eval "$("$CONDA_EXE" shell.bash hook 2>/dev/null)" || true
    else
        for conda_sh in \
            "$HOME/miniconda3/etc/profile.d/conda.sh" \
            "$HOME/anaconda3/etc/profile.d/conda.sh" \
            "$HOME/.conda/etc/profile.d/conda.sh" \
            "/opt/conda/etc/profile.d/conda.sh"; do
            if [ -f "$conda_sh" ]; then
                source "$conda_sh" 2>/dev/null || true
                break
            fi
        done
    fi

    # Try activating the environment if Conda is available
    if type conda >/dev/null 2>&1; then
        if conda info --envs 2>/dev/null | grep -q -E "^\s*${ENV_NAME}\s|/${ENV_NAME}\s"; then
            echo "Activating Conda environment '$ENV_NAME'..."
            if conda activate "$ENV_NAME" 2>/dev/null; then
                echo "Successfully activated '$ENV_NAME' environment."
                return 0
            else
                echo "Warning: Unable to activate Conda environment '$ENV_NAME'." >&2
            fi
        else
            echo "Notice: Conda environment '$ENV_NAME' was not found." >&2
            echo "Please check if the environment exists or create it using: conda create -n $ENV_NAME" >&2
        fi
    else
        echo "Notice: Conda is not available or not configured in this shell session." >&2
    fi

    return 1
}

activate_spacemap_env

# Prefer the source tree next to this launcher.  The Conda environment may
# contain an editable install of another SPACEMAP checkout; without this
# precedence Python can mix app.py from this folder with common.py from the
# other checkout.
export PYTHONPATH="$SCRIPT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"

# Use an explicitly selected interpreter when supplied; otherwise use the
# Python interpreter active in the caller's environment. This keeps the
# launcher portable across machines, Conda environments and containers.
if [ -n "${SPACEMAP_PYTHON:-}" ]; then
    PYTHON_BIN="$SPACEMAP_PYTHON"
else
    PYTHON_BIN="$(command -v python3 || command -v python || true)"
fi

if [ -z "$PYTHON_BIN" ] || ! "$PYTHON_BIN" -c "import streamlit" >/dev/null 2>&1; then
    echo "SPACEMAP requires a Python environment with Streamlit installed." >&2
    echo "Install with: python -m pip install -e \".[notebook]\"" >&2
    exit 1
fi

# Prevent protobuf implementation warnings
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

# Some sites export a web proxy without a matching NO_PROXY. The AI model
# server runs on this same machine; localhost traffic and its private key
# must never be sent through a proxy.
export NO_PROXY="127.0.0.1,localhost${NO_PROXY:+,$NO_PROXY}"
export no_proxy="127.0.0.1,localhost${no_proxy:+,$no_proxy}"

# Cap the numeric thread pools before Python starts.
#
# The OpenBLAS shipped with numpy in this environment is compiled with
# MAX_THREADS=128.  On a machine with more cores than that OpenBLAS runs out
# of its fixed per-thread buffer table, prints
# "Program is Terminated. Because you tried to allocate too many memory
# regions", and then returns a NULL buffer that its caller dereferences - the
# process dies with SIGSEGV.  MiniBatchKMeans in the pixel-classification step
# triggers this reliably, because scikit-learn opens an OpenMP region with one
# thread per core and every one of those threads calls into BLAS.
#
# These must be exported before the interpreter loads numpy: each library reads
# its thread count once, at load time.  32 keeps the OpenMP and BLAS pools
# combined well under the 128 limit, is plenty for a 44-feature k-means, and
# leaves cores free on a shared node.  Override by exporting any of these first.
_SPACEMAP_CORES="$(nproc 2>/dev/null || echo 8)"
_SPACEMAP_THREAD_CAP=$(( _SPACEMAP_CORES < 32 ? _SPACEMAP_CORES : 32 ))
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-$_SPACEMAP_THREAD_CAP}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-$_SPACEMAP_THREAD_CAP}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-$_SPACEMAP_THREAD_CAP}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-$_SPACEMAP_THREAD_CAP}"
echo "Thread pools capped at $_SPACEMAP_THREAD_CAP (node has $_SPACEMAP_CORES cores)."

# server.py runs the same interface and also serves the image viewer's data on
# the same port. Streamlit releases without st.App fall back to app.py; the
# viewer then serves its data on a second port.
_SPACEMAP_ENTRY="$SCRIPT_DIR/src/spacemap_gui/server.py"
if ! "$PYTHON_BIN" -c "import streamlit, sys; sys.exit(0 if hasattr(streamlit, 'App') else 1)" >/dev/null 2>&1; then
    _SPACEMAP_ENTRY="$SCRIPT_DIR/src/spacemap_gui/app.py"
fi

echo "Starting SPACEMAP AI Streamlit GUI..."
exec "$PYTHON_BIN" -m spacemap_gui.launch "$_SPACEMAP_ENTRY" "$@"
