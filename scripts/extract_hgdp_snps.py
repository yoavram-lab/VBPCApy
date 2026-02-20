from __future__ import annotations

import argparse
import logging
from glob import glob
from pathlib import Path
from typing import Iterable

import numpy as np
import scipy.sparse as sp
from cyvcf2 import VCF


logger = logging.getLogger(__name__)


def _iter_vcfs(pattern: str) -> Iterable[Path]:
    paths = sorted(Path(p) for p in glob(pattern))
    if not paths:
        msg = f"No VCF files matched pattern: {pattern}"
        raise FileNotFoundError(msg)
    return paths


def extract_snps_to_npz(vcf_glob: str, out_prefix: Path, *, log_every: int = 1000) -> None:
    paths = list(_iter_vcfs(vcf_glob))
    logger.info("Found %d VCF files", len(paths))

    # Establish sample order from the first VCF
    first_reader = VCF(str(paths[0]))
    samples = list(first_reader.samples)
    n_samples = len(samples)
    first_reader.close()

    data_vals: list[float] = []
    data_rows: list[int] = []
    data_cols: list[int] = []

    mask_rows: list[int] = []
    mask_cols: list[int] = []

    col_names: list[str] = []
    col_idx = 0

    for vcf_path in paths:
        logger.info("Processing %s", vcf_path)
        reader = VCF(str(vcf_path))
        if list(reader.samples) != samples:
            msg = f"Sample order mismatch in {vcf_path}"
            raise ValueError(msg)

        for variant in reader:
            if not variant.is_snp:
                continue
            if len(variant.ALT) != 1 or len(variant.REF) != 1:
                continue

            g = variant.genotype.array()[:, :2]
            missing_any = (g < 0).any(axis=1)
            g_clipped = np.where(g < 0, 0, g)
            g_sum = g_clipped.sum(axis=1)

            rows = np.nonzero(~missing_any)[0]
            if rows.size == 0:
                continue

            vals = g_sum[rows].astype(np.float32, copy=False)

            data_rows.extend(rows.tolist())
            data_cols.extend([col_idx] * rows.size)
            data_vals.extend(vals.tolist())

            mask_rows.extend(rows.tolist())
            mask_cols.extend([col_idx] * rows.size)

            col_names.append(f"CH{variant.CHROM}-POS{variant.POS}-{variant.ID}")
            col_idx += 1

            if log_every > 0 and col_idx % log_every == 0:
                logger.info("Columns so far: %d", col_idx)

        reader.close()

    n_cols = col_idx
    logger.info("Building sparse matrices: n_samples=%d, n_cols=%d, nnz=%d", n_samples, n_cols, len(data_vals))

    data_mx = sp.coo_matrix(
        (np.array(data_vals, dtype=np.float32), (np.array(data_rows), np.array(data_cols))),
        shape=(n_samples, n_cols),
    ).tocsr()

    mask_mx = sp.coo_matrix(
        (np.ones(len(mask_rows), dtype=np.uint8), (np.array(mask_rows), np.array(mask_cols))),
        shape=(n_samples, n_cols),
    ).tocsr()

    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    data_path = out_prefix.with_suffix(".npz")
    mask_path = out_prefix.with_name(f"{out_prefix.stem}_mask.npz")

    sp.save_npz(data_path, data_mx)
    sp.save_npz(mask_path, mask_mx)

    cols_path = out_prefix.with_name(f"{out_prefix.stem}_columns.txt")
    cols_path.write_text("\n".join(col_names))

    logger.info("Wrote data to %s", data_path)
    logger.info("Wrote mask to %s", mask_path)
    logger.info("Wrote column names to %s", cols_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract SNP-only matrix to CSR NPZ + mask.")
    parser.add_argument("--vcf-glob", default="unphased_all_vcf/*.vcf.gz", help="Glob for input VCFs")
    parser.add_argument("--out-prefix", type=Path, default=Path("tools/datasets/HGDP_Edge_2017_snp"), help="Output prefix for NPZ and mask")
    parser.add_argument("--log-every", type=int, default=1000, help="Log every N columns (0 to disable)")
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = parse_args()
    extract_snps_to_npz(args.vcf_glob, args.out_prefix, log_every=args.log_every)


if __name__ == "__main__":
    main()
