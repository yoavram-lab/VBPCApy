#!/usr/bin/env bash
#SBATCH --job-name=vbpca-convergence-detector
#SBATCH --partition=shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=12G
#SBATCH --time=36:00:00
#SBATCH --output=slurm-%x-%A_%a.out

set -Eeuo pipefail

: "${VBPCA_REPO_ROOT:?export the pinned VBPCApy checkout}"
: "${VBPCA_PYTHON:?export the analysis Python executable}"
: "${VBPCA_DETECTOR_MANIFEST:?export the immutable detector manifest}"
: "${VBPCA_DETECTOR_OUTPUT_DIR:?export the detector output directory}"
: "${VBPCA_REVISION:?export the pinned VBPCApy revision}"
: "${VBPCA_DETECTOR_MANIFEST_SHA256:?export the pinned manifest SHA-256}"
: "${SLURM_ARRAY_TASK_ID:?submit this worker with --array}"

if [[ "$(git -C "${VBPCA_REPO_ROOT}" rev-parse HEAD)" != "${VBPCA_REVISION}" ]]; then
    printf 'VBPCApy revision differs from pinned revision %s\n' \
        "${VBPCA_REVISION}" >&2
    exit 2
fi
if [[ -n "$(git -C "${VBPCA_REPO_ROOT}" status --porcelain --untracked-files=no)" ]]; then
    printf 'VBPCApy checkout has tracked changes.\n' >&2
    exit 2
fi
if [[ "$(sha256sum "${VBPCA_DETECTOR_MANIFEST}" | cut -d ' ' -f 1)" != "${VBPCA_DETECTOR_MANIFEST_SHA256}" ]]; then
    printf 'Detector manifest checksum mismatch.\n' >&2
    exit 2
fi

export PYTHONPATH="${VBPCA_REPO_ROOT}/src:${VBPCA_REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
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

"${VBPCA_PYTHON}" -m analysis.trade_study.convergence_stability_study run-shard \
    --manifest "${VBPCA_DETECTOR_MANIFEST}" \
    --shard-index "${SLURM_ARRAY_TASK_ID}" \
    --retry-errors \
    --output-dir "${VBPCA_DETECTOR_OUTPUT_DIR}"
