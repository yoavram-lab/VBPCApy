#!/usr/bin/env bash
#SBATCH --job-name=vbpca-backend-292
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --hint=nomultithread
#SBATCH --mem=4G
#SBATCH --time=00:10:00
#SBATCH --no-requeue
#SBATCH --output=slurm-%x-%j.out
set -Eeuo pipefail
: "${VBPCA_PROFILE_ROOT:?export the new staged profile directory}"
: "${VBPCA_PROFILE_PYTHON:?export the isolated pinned Python interpreter}"
: "${VBPCA_PROFILE_REGISTRATION_SHA256:?export registration checksum}"

export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export BLIS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
unset PYTHONPATH
cd -- "${VBPCA_PROFILE_ROOT}"
sha256sum --check archives.sha256
"${VBPCA_PROFILE_PYTHON}" candidate/analysis/cluster/prepare_backend_comparison.py \
    --root "${VBPCA_PROFILE_ROOT}" \
    --registration-sha256 "${VBPCA_PROFILE_REGISTRATION_SHA256}"
read -r -a profile_cpus < cpu-ids.txt
"${VBPCA_PROFILE_PYTHON}" candidate/analysis/compare_backend_revisions.py \
    --baseline "${VBPCA_PROFILE_ROOT}/baseline" \
    --candidate "${VBPCA_PROFILE_ROOT}/candidate" \
    --cpus "${profile_cpus[@]}" --repeats 3 \
    --output "${VBPCA_PROFILE_ROOT}/results"
