#!/usr/bin/env bash
#SBATCH --job-name=vbpca-defaults-scale
#SBATCH --partition=parallel
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=24
#SBATCH --mem=192000M
#SBATCH --time=72:00:00
#SBATCH --output=slurm-%x-%A_%a.out

set -Eeuo pipefail

: "${VBPCA_REPO_ROOT:?export the pinned VBPCApy checkout}"
: "${VBPCA_PYTHON:?export the analysis Python executable}"
: "${VBPCA_DEFAULTS_MANIFEST:?export the immutable manifest path}"
: "${VBPCA_DEFAULTS_OUTPUT_DIR:?export the shard-results directory}"
: "${VBPCA_REVISION:?export the pinned VBPCApy revision}"
: "${VBPCA_DEFAULTS_MANIFEST_SHA256:?export the pinned manifest SHA-256}"
: "${VBPCA_RELEASE:?export the pinned VBPCA release}"
: "${VBPCA_DEFAULTS_SHARD_COUNT:?export the manifest shard count}"
: "${SLURM_ARRAY_TASK_ID:?submit this worker as a job array}"

if [[ ! "${VBPCA_DEFAULTS_SHARD_COUNT}" =~ ^[1-9][0-9]*$ ]]; then
    printf 'VBPCA_DEFAULTS_SHARD_COUNT must be a positive integer.\n' >&2
    exit 2
fi
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
actual_manifest_sha256="$(sha256sum "${VBPCA_DEFAULTS_MANIFEST}" | cut -d ' ' -f 1)"
if [[ "${actual_manifest_sha256}" != "${VBPCA_DEFAULTS_MANIFEST_SHA256}" ]]; then
    printf 'Manifest checksum differs from pinned SHA-256 %s\n' \
        "${VBPCA_DEFAULTS_MANIFEST_SHA256}" >&2
    exit 2
fi

export PYTHONPATH="${VBPCA_REPO_ROOT}/src:${VBPCA_REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONUNBUFFERED="1"
export OMP_NUM_THREADS="1"
export OPENBLAS_NUM_THREADS="1"
export MKL_NUM_THREADS="1"
export BLIS_NUM_THREADS="1"
export NUMEXPR_NUM_THREADS="1"

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

base_index=$((2 * SLURM_ARRAY_TASK_ID))
if ((base_index >= VBPCA_DEFAULTS_SHARD_COUNT)); then
    printf 'Array task %s starts beyond %s shards.\n' \
        "${SLURM_ARRAY_TASK_ID}" "${VBPCA_DEFAULTS_SHARD_COUNT}" >&2
    exit 2
fi
indices=("${base_index}")
if ((base_index + 1 < VBPCA_DEFAULTS_SHARD_COUNT)); then
    indices+=("$((base_index + 1))")
fi

printf 'VBPCApy release: %s\nVBPCApy revision: %s\nManifest SHA-256: %s\nShards: %s\n' \
    "${VBPCA_RELEASE}" "${VBPCA_REVISION}" "${actual_manifest_sha256}" \
    "${indices[*]}"

pids=()
for shard_index in "${indices[@]}"; do
    srun --exclusive --exact --nodes=1 --ntasks=1 --cpus-per-task=24 \
        --cpu-bind=cores \
        env VBPCA_NUM_THREADS=24 \
        "${VBPCA_PYTHON}" -m analysis.trade_study.modern_defaults_study \
        run-shard \
        --manifest "${VBPCA_DEFAULTS_MANIFEST}" \
        --shard-index "${shard_index}" \
        --output-dir "${VBPCA_DEFAULTS_OUTPUT_DIR}" \
        --num-cpu 24 &
    pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
    if ! wait "${pid}"; then
        status=1
    fi
done
exit "${status}"
