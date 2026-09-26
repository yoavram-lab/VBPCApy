#!/usr/bin/env bash
#SBATCH --job-name=vbpca-residual-cap
#SBATCH --partition=shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=24:00:00
#SBATCH --output=slurm-%x-%A_%a.out

set -Eeuo pipefail

: "${VBPCA_REPO_ROOT:?export the pinned VBPCApy checkout}"
: "${VBPCA_TRADE_STUDY_ROOT:?export the pinned trade-study checkout}"
: "${VBPCA_PYTHON:?export the analysis Python executable}"
: "${VBPCA_RESIDUAL_MANIFEST:?export the immutable manifest path}"
: "${VBPCA_RESIDUAL_OUTPUT_DIR:?export the shard-results directory}"
: "${VBPCA_REVISION:?export the pinned VBPCApy revision}"
: "${VBPCA_TRADE_STUDY_REVISION:?export the pinned trade-study revision}"
: "${VBPCA_RESIDUAL_MANIFEST_SHA256:?export the pinned manifest SHA-256}"
: "${VBPCA_RELEASE:?export the pinned VBPCA release}"
: "${SLURM_ARRAY_TASK_ID:?submit this worker with --array=0-20}"

if [[ ! -x "${VBPCA_PYTHON}" ]]; then
    printf 'Python executable not found: %s\n' "${VBPCA_PYTHON}" >&2
    exit 2
fi
if [[ "$(git -C "${VBPCA_REPO_ROOT}" rev-parse HEAD)" != "${VBPCA_REVISION}" ]]; then
    printf 'VBPCApy revision differs from pinned revision %s\n' \
        "${VBPCA_REVISION}" >&2
    exit 2
fi
if [[ -n "$(git -C "${VBPCA_REPO_ROOT}" status --porcelain --untracked-files=no)" ]]; then
    printf 'VBPCApy checkout has tracked changes.\n' >&2
    exit 2
fi
if [[ "$(git -C "${VBPCA_TRADE_STUDY_ROOT}" rev-parse HEAD)" != "${VBPCA_TRADE_STUDY_REVISION}" ]]; then
    printf 'trade-study revision differs from pinned revision %s\n' \
        "${VBPCA_TRADE_STUDY_REVISION}" >&2
    exit 2
fi
if [[ -n "$(git -C "${VBPCA_TRADE_STUDY_ROOT}" status --porcelain --untracked-files=no)" ]]; then
    printf 'trade-study checkout has tracked changes.\n' >&2
    exit 2
fi
actual_manifest_sha256="$(sha256sum "${VBPCA_RESIDUAL_MANIFEST}" | cut -d ' ' -f 1)"
if [[ "${actual_manifest_sha256}" != "${VBPCA_RESIDUAL_MANIFEST_SHA256}" ]]; then
    printf 'Manifest checksum differs from pinned SHA-256 %s\n' \
        "${VBPCA_RESIDUAL_MANIFEST_SHA256}" >&2
    exit 2
fi

export PYTHONPATH="${VBPCA_REPO_ROOT}/src:${VBPCA_TRADE_STUDY_ROOT}/src:${VBPCA_REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONUNBUFFERED="1"
export OMP_NUM_THREADS="1"
export OPENBLAS_NUM_THREADS="1"
export MKL_NUM_THREADS="1"
export BLIS_NUM_THREADS="1"
export NUMEXPR_NUM_THREADS="1"
export VBPCA_NUM_THREADS="1"

cd -- "${VBPCA_REPO_ROOT}"
vbpca_import="$("${VBPCA_PYTHON}" -c 'from pathlib import Path; import vbpca_py; print(Path(vbpca_py.__file__).resolve())')"
case "${vbpca_import}" in
    "${VBPCA_REPO_ROOT}/src"/*) ;;
    *)
        printf 'VBPCApy imported from %s, expected below %s/src\n' \
            "${vbpca_import}" "${VBPCA_REPO_ROOT}" >&2
        exit 2
        ;;
esac
actual_release="$("${VBPCA_PYTHON}" -c 'import vbpca_py; print(vbpca_py.__version__)')"
if [[ "${actual_release}" != "${VBPCA_RELEASE}" ]]; then
    printf 'VBPCA release %s differs from pinned release %s\n' \
        "${actual_release}" "${VBPCA_RELEASE}" >&2
    exit 2
fi
trade_study_import="$("${VBPCA_PYTHON}" -c 'from pathlib import Path; import trade_study; print(Path(trade_study.__file__).resolve())')"
case "${trade_study_import}" in
    "${VBPCA_TRADE_STUDY_ROOT}/src"/*) ;;
    *)
        printf 'trade-study imported from %s, expected below %s/src\n' \
            "${trade_study_import}" "${VBPCA_TRADE_STUDY_ROOT}" >&2
        exit 2
        ;;
esac

printf 'VBPCApy release: %s\nVBPCApy revision: %s\ntrade-study revision: %s\n' \
    "${VBPCA_RELEASE}" "${VBPCA_REVISION}" "${VBPCA_TRADE_STUDY_REVISION}"
"${VBPCA_PYTHON}" -c 'import platform, numpy, scipy; print(f"Python: {platform.python_version()}"); print(f"NumPy: {numpy.__version__}"); print(f"SciPy: {scipy.__version__}")'

"${VBPCA_PYTHON}" -m analysis.trade_study.residual_convergence_followup \
    run-shard \
    --manifest "${VBPCA_RESIDUAL_MANIFEST}" \
    --shard-index "${SLURM_ARRAY_TASK_ID}" \
    --output-dir "${VBPCA_RESIDUAL_OUTPUT_DIR}" \
    --n-jobs "${SLURM_CPUS_PER_TASK:-1}"
