# Plot_Bands.py
# Usage:
#   python Plot_Bands.py <seed>
#
# Example:
#   python Plot_Bands.py Ca3Mn2O7_X2_m1_X3_m1

# This version:
#   1. Uses CASTEPbands to generate the automatic k-path labels.
#   2. Prints the detected path labels.
#   3. Optionally asks whether you want to manually override them.
#   4. Keeps switches for gap marker and gap text box.
#   5. Allows easy control of axis labels and font sizes.
#
# Outputs:
#   <seed>_both.png
#   <seed>_Up.png
#   <seed>_Down.png
#   <seed>_Up.dat
#   <seed>_Down.dat
#   <seed>_kpoints.dat

import sys
import numpy as np
import matplotlib

# Useful for running on HPC/login nodes without a display
matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib import rcParams
from CASTEPbands import Spectral


# ============================================================
# USER INPUT
# ============================================================

if len(sys.argv) < 2:
    print("Usage: python Plot_Bands.py <seed>")
    sys.exit(1)

seed = sys.argv[1]

# Energy window
ymin = -5.0
ymax = 0.0

# Figure size
fig_w = 7
fig_h = 5

# Gap switches
show_gap_text = False        # Top-left gap text box
show_gap_marker = False      # CASTEPbands gap arrows/markers on the both-spin plot

# K-label behaviour
# True:
#   Print CASTEPbands detected path and ask whether to override manually.
# False:
#   Always use CASTEPbands automatic labels without asking.
interactive_klabels = True

# Font sizes
title_size = 15
xlabel_size = 22
ylabel_size = 24
xtick_size = 20
ytick_size = 20
gap_text_size = 10

# Axis labels
x_axis_label = ""
y_axis_label = r"$E - E_F$ (eV)"

# Output quality
dpi = 300


# ============================================================
# MATPLOTLIB SETTINGS
# ============================================================

print(f"\nRUNNING SCRIPT FOR {seed}\n")

rcParams.update({
    "text.usetex": False,
    "font.family": "serif",

    "axes.linewidth": 1.5,

    "xtick.direction": "in",
    "ytick.direction": "in",

    "xtick.top": False,
    "ytick.right": True,

    "ytick.left": True,
    "ytick.minor.visible": True,

    "xtick.major.size": 7,
    "ytick.major.size": 7,
    "ytick.minor.size": 5,
})


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def clean_detected_label(label):
    """
    Make detected matplotlib labels easier to read in the terminal.

    This is only for printing. It does not control the actual plot labels.
    """
    label = str(label).strip()

    # Remove common mathtext wrappers
    label = label.replace("$", "")
    label = label.replace("\\Gamma", "Gamma")
    label = label.replace("Γ", "Gamma")

    return label


def label_to_mathtext(label):
    """
    Convert user input labels into matplotlib mathtext labels.

    Examples:
        Gamma -> $\\Gamma$
        G     -> $\\Gamma$
        Γ     -> $\\Gamma$
        R     -> $R$
        R'    -> $R'$
    """
    label = label.strip()
    label = label.replace(" ", "")

    lower = label.lower()

    if lower in ["gamma", "gam", "g", "Γ".lower(), r"\gamma"]:
        return r"$\Gamma$"

    if label in [r"\Gamma", "Γ"]:
        return r"$\Gamma$"

    return rf"${label}$"


def parse_manual_labels(user_input):
    """
    Parse manual labels from user input.

    Accepted examples:
        R Gamma R'
        R, Gamma, R'
        T Gamma Y Z L R Gamma
    """
    user_input = user_input.strip()

    if "," in user_input:
        parts = [x.strip() for x in user_input.split(",") if x.strip()]
    else:
        parts = [x.strip() for x in user_input.split() if x.strip()]

    return [label_to_mathtext(x) for x in parts]


def plot_for_label_detection(bs, ax):
    """
    Make a temporary CASTEPbands plot only to read its automatic x-ticks
    and x-labels.

    This avoids editing the CASTEPbands library itself.
    """
    if bs.nspins == 2:
        bs.plot_bs(
            ax,
            spin_polarised=True,
            Elim=(ymin, ymax),
            mark_gap=False
        )
    else:
        bs.plot_bs(
            ax,
            Elim=(ymin, ymax),
            mark_gap=False
        )


def get_default_kpath_from_castepbands(bs):
    """
    Ask CASTEPbands to generate the plot once, then read the high-symmetry
    tick positions and labels from matplotlib.
    """
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    plot_for_label_detection(bs, ax)

    # Force matplotlib to actually populate tick labels
    fig.canvas.draw()

    xticks = ax.get_xticks()
    raw_labels = [tick.get_text() for tick in ax.get_xticklabels()]
    cleaned_labels = [clean_detected_label(x) for x in raw_labels]

    plt.close(fig)

    return xticks, raw_labels, cleaned_labels


def choose_klabels(bs):
    """
    Detect automatic k-path labels using CASTEPbands.

    Logic:
        interactive_klabels = False
            -> always use CASTEPbands automatic labels.

        interactive_klabels = True
            -> print CASTEPbands labels, then ask whether to manually override.
    """
    xticks, raw_labels, cleaned_labels = get_default_kpath_from_castepbands(bs)

    print("Detected k-path from CASTEPbands:")
    print("  Raw matplotlib labels:")
    print(f"  {raw_labels}")
    print("  Cleaned labels:")
    print(f"  {' - '.join(cleaned_labels)}")
    print(f"  Number of high-symmetry points detected: {len(xticks)}\n")

    if not interactive_klabels:
        print("interactive_klabels = False")
        print("Using CASTEPbands automatic k-path labels.\n")
        return xticks, raw_labels

    try:
        answer = input("Do you want to assign manual k-path labels? [y/N]: ").strip().lower()
    except EOFError:
        print("No interactive input detected.")
        print("Using CASTEPbands automatic k-path labels.\n")
        return xticks, raw_labels

    if answer not in ["y", "yes"]:
        print("Using CASTEPbands automatic k-path labels.\n")
        return xticks, raw_labels

    print("\nEnter the labels in order.")
    print("Examples:")
    print("  R Gamma R'")
    print("  X Gamma Y")
    print("  T Gamma Y Z L R Gamma")
    print("You can use either spaces or commas.\n")

    while True:
        try:
            user_input = input("Manual labels: ").strip()
        except EOFError:
            print("No interactive input detected.")
            print("Using CASTEPbands automatic k-path labels.\n")
            return xticks, raw_labels

        new_labels = parse_manual_labels(user_input)

        if len(new_labels) == len(xticks):
            print("\nUsing manual k-path labels:")
            print(new_labels)
            print("")
            return xticks, new_labels

        print("\nLabel number mismatch.")
        print(f"CASTEPbands detected {len(xticks)} high-symmetry points.")
        print(f"You provided {len(new_labels)} labels.")
        print("Please try again.\n")


def format_band_axis(ax, xticks, klabels):
    """
    Apply consistent axis labels, tick sizes, and k-point labels.

    This must be called after bs.plot_bs(), because plot_bs() first creates
    its own automatic ticks and labels.
    """
    ax.set_xlabel(x_axis_label, fontsize=xlabel_size)
    ax.set_ylabel(y_axis_label, fontsize=ylabel_size)

    ax.tick_params(axis="x", which="major", labelsize=xtick_size)
    ax.tick_params(axis="y", which="major", labelsize=ytick_size)

    ax.set_ylim(ymin, ymax)

    if len(xticks) == len(klabels):
        ax.set_xticks(xticks)
        ax.set_xticklabels(klabels, fontsize=xtick_size)
    else:
        print("\nWARNING:")
        print(f"Found {len(xticks)} x-tick positions, but {len(klabels)} labels were provided.")
        print("Keeping CASTEPbands automatic labels.\n")

    return ax


def plot_band_structure(bs, seed, spin_mode, xticks, klabels):
    """
    Plot one band-structure figure.

    spin_mode options:
        "both"
        "up"
        "down"
    """
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    if spin_mode == "both":
        if bs.nspins == 2:
            bs.plot_bs(
                ax,
                spin_polarised=True,
                Elim=(ymin, ymax),
                mark_gap=show_gap_marker
            )
        else:
            bs.plot_bs(
                ax,
                Elim=(ymin, ymax),
                mark_gap=show_gap_marker
            )

        ax.set_title(f"{seed} both", fontsize=title_size)

    elif spin_mode == "up":
        bs.plot_bs(
            ax,
            spin_index=0,
            Elim=(ymin, ymax),
            mark_gap=False
        )

        ax.set_title(f"{seed} up", fontsize=title_size)

    elif spin_mode == "down":
        bs.plot_bs(
            ax,
            spin_index=1,
            Elim=(ymin, ymax),
            mark_gap=False
        )

        ax.set_title(f"{seed} down", fontsize=title_size)

    else:
        raise ValueError(f"Unknown spin_mode: {spin_mode}")

    format_band_axis(ax, xticks, klabels)

    return fig, ax


# ============================================================
# LOAD DATA
# ============================================================

bs = Spectral.Spectral(
    seed,
    zero_fermi=True,
    high_sym_spacegroup=True
)

info = bs.get_band_info(silent=True)

print("Band information:")
print("Keys:", info.keys())
print(f"Seed = {seed}")
print(f"nspins = {bs.nspins}")
print(f"VBM = {info['vbm']}")
print(f"CBM = {info['cbm']}")
print(f"Indirect gap = {info['gap_indir']}")
print(f"Direct gap = {info['gap_dir']}")
print(f"Indirect gap kpts = {info['loc_indir']}")
print(f"Direct gap kpts = {info['loc_dir']}")
print("")


# ============================================================
# CHOOSE K-PATH LABELS
# ============================================================

xticks, final_klabels = choose_klabels(bs)


# ============================================================
# SAVE RAW DATA
# ============================================================

up = bs.BandStructure[:, :, 0]
np.savetxt(f"{seed}_Up.dat", up.T)

if bs.nspins == 2:
    down = bs.BandStructure[:, :, 1]
    np.savetxt(f"{seed}_Down.dat", down.T)

np.savetxt(f"{seed}_kpoints.dat", bs.kpoints)

print("Saved raw band data.\n")


# ============================================================
# GAP TEXT
# ============================================================

if bs.nspins == 2:
    gap_up = info["gap_indir"][0]
    gap_down = info["gap_indir"][1]
    gap_global = min(info["gap_indir"])

    gap_text = (
        f"Gap_up   = {gap_up:.6f} eV\n"
        f"Gap_down = {gap_down:.6f} eV\n"
        f"Gap_min  = {gap_global:.6f} eV"
    )
else:
    gap_text = f"Gap = {info['gap_indir'][0]:.6f} eV"


# ============================================================
# BOTH SPINS
# ============================================================

fig, ax = plot_band_structure(
    bs=bs,
    seed=seed,
    spin_mode="both",
    xticks=xticks,
    klabels=final_klabels
)

if show_gap_text:
    ax.text(
        0.02, 0.98,
        gap_text,
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=gap_text_size,
        bbox=dict(facecolor="white", alpha=0.8)
    )

plt.tight_layout()
plt.savefig(f"{seed}_both.png", dpi=dpi)
plt.close(fig)


# ============================================================
# UP ONLY
# ============================================================

fig, ax = plot_band_structure(
    bs=bs,
    seed=seed,
    spin_mode="up",
    xticks=xticks,
    klabels=final_klabels
)

plt.tight_layout()
plt.savefig(f"{seed}_Up.png", dpi=dpi)
plt.close(fig)


# ============================================================
# DOWN ONLY
# ============================================================

if bs.nspins == 2:
    fig, ax = plot_band_structure(
        bs=bs,
        seed=seed,
        spin_mode="down",
        xticks=xticks,
        klabels=final_klabels
    )

    plt.tight_layout()
    plt.savefig(f"{seed}_Down.png", dpi=dpi)
    plt.close(fig)


# ============================================================
# FINISH
# ============================================================

print("Saved plots:")
print(f"  {seed}_both.png")
print(f"  {seed}_Up.png")

if bs.nspins == 2:
    print(f"  {seed}_Down.png")
