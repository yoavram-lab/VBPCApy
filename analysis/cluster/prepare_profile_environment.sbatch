#!/usr/bin/env bash
#SBATCH --job-name=vbpca-profile-env
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:10:00
#SBATCH --no-requeue
#SBATCH --output=slurm-%x-%j.out
set -Eeuo pipefail
: "${VBPCA_PROFILE_ENV:?export a new isolated environment path}"
test ! -e "${VBPCA_PROFILE_ENV}"
/usr/bin/python3.11 -m venv "${VBPCA_PROFILE_ENV}"
"${VBPCA_PROFILE_ENV}/bin/python" -m pip install --only-binary=:all: \
    'numpy==2.2.6' 'scipy==1.17.1' 'threadpoolctl==3.7.0' 'vbpca_py==0.4.10'
"${VBPCA_PROFILE_ENV}/bin/python" -m pip freeze
