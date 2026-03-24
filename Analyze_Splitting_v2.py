import argparse
import csv
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from CASTEPbands import Spectral


def parse_args():
    p = argparse.ArgumentParser(
        description="Analyse spin splitting from spin-polarised CASTEP .bands files."
    )
    p.add_argument("--prefix", default="AM_{u}_L",
                   help="Seed pattern. Use {u} where the Hubbard U integer should go.")
    p.add_argument("--u-min", type=int, default=0)
    p.add_argument("--u-max", type=int, default=9)
    p.add_argument("--emin", type=float, default=-2.0,
                   help="Minimum E-EF (eV) included in splitting analysis.")
    p.add_argument("--emax", type=float, default=0.5,
                   help="Maximum E-EF (eV) included in splitting analysis.")
    p.add_argument("--percentile", type=float, default=95.0,
                   help="Percentile used for a robust splitting metric.")
    p.add_argument("--topn", type=int, default=10,
                   help="Average of the top N absolute splittings.")
    p.add_argument("--out-prefix", default="splitting_summary",
                   help="Prefix for CSV and figures.")
    return p.parse_args()


def get_gap_scalar(info):
    gap = info.get("gap_indir", np.nan)
    arr = np.asarray(gap, dtype=float)
    if arr.ndim == 0:
        return float(arr)
    return float(np.nanmin(arr))


def get_k_info(bs, k_idx):
    """Handle both 1D path-coordinate k arrays and 2D/3D k-point arrays."""
    kraw = np.asarray(bs.kpoints)

    # Case 1: bs.kpoints is a simple 1D path coordinate array.
    if kraw.ndim == 1:
        return float(kraw[k_idx]), np.nan, np.nan, "path_coordinate"

    # Case 2: bs.kpoints is shape (nk,3) or similar.
    if kraw.ndim >= 2:
        row = np.asarray(kraw[k_idx], dtype=float).ravel()
        if row.size >= 3:
            return float(row[0]), float(row[1]), float(row[2]), "kxkykz"
        if row.size == 2:
            return float(row[0]), float(row[1]), np.nan, "k1k2"
        if row.size == 1:
            return float(row[0]), np.nan, np.nan, "path_coordinate"

    # Fallback.
    return np.nan, np.nan, np.nan, "unknown"


def analyse_seed(seed, emin, emax, percentile, topn):
    bs = Spectral.Spectral(seed, zero_fermi=True, high_sym_spacegroup=True)
    info = bs.get_band_info(silent=True)

    if bs.nspins != 2:
        raise ValueError(f"{seed}: expected nspins = 2, got {bs.nspins}")

    bands = np.asarray(bs.BandStructure, dtype=float)
    if bands.ndim != 3 or bands.shape[2] != 2:
        raise ValueError(f"{seed}: expected BandStructure shape [nband, nk, 2], got {bands.shape}")

    up = bands[:, :, 0]
    down = bands[:, :, 1]
    delta = up - down
    abs_delta = np.abs(delta)

    # Include a point if either spin branch lies in the chosen energy window.
    mask = ((up >= emin) & (up <= emax)) | ((down >= emin) & (down <= emax))

    selected_abs = abs_delta[mask]
    if selected_abs.size == 0:
        return {
            "seed": seed,
            "gap_eV": get_gap_scalar(info),
            "n_selected_points": 0,
            "max_abs_split_eV": np.nan,
            "signed_split_at_max_eV": np.nan,
            "robust_percentile_split_eV": np.nan,
            "topn_mean_split_eV": np.nan,
            "band_index_at_max": np.nan,
            "k_index_at_max": np.nan,
            "k_x": np.nan,
            "k_y": np.nan,
            "k_z": np.nan,
            "k_repr": "none",
            "energy_up_at_max_eV": np.nan,
            "energy_down_at_max_eV": np.nan,
            "emin_window_eV": emin,
            "emax_window_eV": emax,
        }

    masked_abs = np.where(mask, abs_delta, -np.inf)
    flat_idx = int(np.argmax(masked_abs))
    band_idx, k_idx = np.unravel_index(flat_idx, abs_delta.shape)

    topn_eff = min(topn, selected_abs.size)
    topn_vals = np.sort(selected_abs)[-topn_eff:]

    kx, ky, kz, k_repr = get_k_info(bs, k_idx)

    return {
        "seed": seed,
        "gap_eV": get_gap_scalar(info),
        "n_selected_points": int(selected_abs.size),
        "max_abs_split_eV": float(abs_delta[band_idx, k_idx]),
        "signed_split_at_max_eV": float(delta[band_idx, k_idx]),
        "robust_percentile_split_eV": float(np.percentile(selected_abs, percentile)),
        "topn_mean_split_eV": float(np.mean(topn_vals)),
        "band_index_at_max": int(band_idx),
        "k_index_at_max": int(k_idx),
        "k_x": kx,
        "k_y": ky,
        "k_z": kz,
        "k_repr": k_repr,
        "energy_up_at_max_eV": float(up[band_idx, k_idx]),
        "energy_down_at_max_eV": float(down[band_idx, k_idx]),
        "emin_window_eV": emin,
        "emax_window_eV": emax,
    }


def save_csv(rows, filename):
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with open(filename, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def make_plot(rows, out_png):
    u_vals, gap_vals, max_vals, pct_vals, topn_vals = [], [], [], [], []
    for row in rows:
        seed = row["seed"]
        try:
            u = int(seed.split("_")[1])
        except Exception:
            u = len(u_vals)
        u_vals.append(u)
        gap_vals.append(row["gap_eV"])
        max_vals.append(row["max_abs_split_eV"])
        pct_vals.append(row["robust_percentile_split_eV"])
        topn_vals.append(row["topn_mean_split_eV"])

    plt.figure(figsize=(7, 5))
    plt.plot(u_vals, gap_vals, marker="o", label="Band Gap")
    plt.plot(u_vals, max_vals, marker="s", label="Max |ΔE|")
    plt.plot(u_vals, pct_vals, marker="^", label="Percentile |ΔE|")
    plt.plot(u_vals, topn_vals, marker="d", label="Top-N mean |ΔE|")
    plt.xlabel("Hubbard U (eV)")
    plt.ylabel("Energy (eV)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.close()


def main():
    args = parse_args()
    rows = []

    for u in range(args.u_min, args.u_max + 1):
        seed = args.prefix.format(u=u)
        if not Path(f"{seed}.bands").exists():
            print(f"Skipping {seed}: no .bands file found")
            continue
        try:
            row = analyse_seed(seed, args.emin, args.emax, args.percentile, args.topn)
            rows.append(row)
            print(
                f"{seed}: gap={row['gap_eV']:.6f} eV, "
                f"max|ΔE|={row['max_abs_split_eV']:.6f} eV, "
                f"p{args.percentile:g}={row['robust_percentile_split_eV']:.6f} eV, "
                f"topN={row['topn_mean_split_eV']:.6f} eV, "
                f"band={row['band_index_at_max']}, k_idx={row['k_index_at_max']}, "
                f"k=({row['k_x']}, {row['k_y']}, {row['k_z']}) [{row['k_repr']}]"
            )
        except Exception as exc:
            print(f"Failed on {seed}: {exc}")

    if not rows:
        raise SystemExit("No valid seeds were analysed.")

    csv_name = f"{args.out_prefix}.csv"
    png_name = f"{args.out_prefix}.png"
    save_csv(rows, csv_name)
    make_plot(rows, png_name)
    print(f"Saved {csv_name}")
    print(f"Saved {png_name}")


if __name__ == "__main__":
    main()

