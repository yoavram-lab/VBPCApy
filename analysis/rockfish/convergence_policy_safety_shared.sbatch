#!/usr/bin/env bash
#SBATCH --job-name=vbpca-policy-safety
#SBATCH --partition=shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=12G
#SBATCH --time=36:00:00
#SBATCH --output=slurm-%x-%A_%a.out

set -Eeuo pipefail

: "${VBPCA_REPO_ROOT:?export the pinned VBPCApy checkout}"
: "${PP_EIGENTEST_REPO_ROOT:?export the pinned pp-eigentest checkout}"
: "${VBPCA_PYTHON:?export the analysis Python executable}"
: "${VBPCA_POLICY_SAFETY_MANIFEST:?export the immutable safety manifest}"
: "${VBPCA_POLICY_SAFETY_OUTPUT_DIR:?export the result directory}"
: "${VBPCA_REVISION:?export the pinned VBPCApy revision}"
: "${PP_EIGENTEST_REVISION:?export the pinned pp-eigentest revision}"
: "${VBPCA_POLICY_SAFETY_MANIFEST_SHA256:?export the manifest SHA-256}"
: "${SLURM_ARRAY_TASK_ID:?submit this worker with --array}"

if [[ "$(git -C "${VBPCA_REPO_ROOT}" rev-parse HEAD)" != "${VBPCA_REVISION}" ]]; then
    printf 'VBPCApy revision differs from pinned revision %s\n' \
        "${VBPCA_REVISION}" >&2
    exit 2
fi
if [[ "$(git -C "${PP_EIGENTEST_REPO_ROOT}" rev-parse HEAD)" != "${PP_EIGENTEST_REVISION}" ]]; then
    printf 'pp-eigentest revision differs from pinned revision %s\n' \
        "${PP_EIGENTEST_REVISION}" >&2
    exit 2
fi
if [[ -n "$(git -C "${VBPCA_REPO_ROOT}" status --porcelain --untracked-files=no)" ]]; then
    printf 'VBPCApy checkout has tracked changes.\n' >&2
    exit 2
fi
if [[ -n "$(git -C "${PP_EIGENTEST_REPO_ROOT}" status --porcelain --untracked-files=no)" ]]; then
    printf 'pp-eigentest checkout has tracked changes.\n' >&2
    exit 2
fi
if [[ "$(sha256sum "${VBPCA_POLICY_SAFETY_MANIFEST}" | cut -d ' ' -f 1)" != "${VBPCA_POLICY_SAFETY_MANIFEST_SHA256}" ]]; then
    printf 'Convergence-policy safety manifest checksum mismatch.\n' >&2
    exit 2
fi

export PYTHONPATH="${VBPCA_REPO_ROOT}/src:${VBPCA_REPO_ROOT}:${PP_EIGENTEST_REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONUNBUFFERED="1"
export OMP_NUM_THREADS="1"
export OPENBLAS_NUM_THREADS="1"
export MKL_NUM_THREADS="1"
export BLIS_NUM_THREADS="1"
export NUMEXPR_NUM_THREADS="1"
export VBPCA_NUM_THREADS="1"
export PP_EIGENTEST_AUTOBENCH="0"

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
pp_import="$("${VBPCA_PYTHON}" -c 'from pathlib import Path; import pp_eigentest; print(Path(pp_eigentest.__file__).resolve())')"
case "${pp_import}" in
    "${PP_EIGENTEST_REPO_ROOT}/src"/*) ;;
    *)
        printf 'pp-eigentest imported from %s, expected below %s/src\n' \
            "${pp_import}" "${PP_EIGENTEST_REPO_ROOT}" >&2
        exit 2
        ;;
esac

"${VBPCA_PYTHON}" -m analysis.trade_study.convergence_policy_safety run-shard \
    --manifest "${VBPCA_POLICY_SAFETY_MANIFEST}" \
    --shard-index "${SLURM_ARRAY_TASK_ID}" \
    --retry-errors \
    --output-dir "${VBPCA_POLICY_SAFETY_OUTPUT_DIR}"
