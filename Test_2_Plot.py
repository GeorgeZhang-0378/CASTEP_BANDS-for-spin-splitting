import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from CASTEPbands import Spectral

# ========= USER INPUT =========
seed = "AM_0"      # change this each time
ymin = -15.0
ymax = 25.0
fig_w = 7
fig_h = 5
# ==============================

print("RUNNING CLEAN SCRIPT")

rcParams.update({
    "text.usetex": False,
    "font.family": "serif",
    "ytick.direction": "in",
    "ytick.left": True,
    "ytick.right": True,
    "ytick.minor.visible": True,
    "ytick.major.size": 7,
    "ytick.minor.size": 5,
})

# ======================
# LOAD DATA
# ======================
bs = Spectral.Spectral(
    seed,
    zero_fermi=True,
    high_sym_spacegroup=True
)

info = bs.get_band_info(silent=True)

print("Keys:", info.keys())
print(f"Seed = {seed}")
print(f"nspins = {bs.nspins}")
print(f"VBM = {info['vbm']}")
print(f"CBM = {info['cbm']}")
print(f"Indirect gap = {info['gap_indir']}")
print(f"Direct gap = {info['gap_dir']}")
print(f"Indirect gap kpts = {info['loc_indir']}")
print(f"Direct gap kpts = {info['loc_dir']}")

# ======================
# SAVE RAW DATA
# ======================
up = bs.BandStructure[:, :, 0]
np.savetxt(f"{seed}_Up.dat", up.T)

if bs.nspins == 2:
    down = bs.BandStructure[:, :, 1]
    np.savetxt(f"{seed}_Down.dat", down.T)

np.savetxt(f"{seed}_kpoints.dat", bs.kpoints)

print("Saved raw band data.")

# ======================
# GAP TEXT
# ======================
if bs.nspins == 2:
    gap_up = info['gap_indir'][0]
    gap_down = info['gap_indir'][1]
    gap_global = min(info['gap_indir'])

    gap_text = (
        f"Gap_up   = {gap_up:.6f} eV\n"
        f"Gap_down = {gap_down:.6f} eV\n"
        f"Gap_min  = {gap_global:.6f} eV"
    )
else:
    gap_text = f"Gap = {info['gap_indir'][0]:.6f} eV"

# ======================
# 1️⃣ BOTH SPINS
# ======================
fig, ax = plt.subplots(figsize=(fig_w, fig_h))

bs.plot_bs(
    ax,
    spin_polarised=True,
    Elim=(ymin, ymax),
    mark_gap=True
)

ax.text(
    0.02, 0.98,
    gap_text,
    transform=ax.transAxes,
    va="top",
    ha="left",
    fontsize=10,
    bbox=dict(facecolor="white", alpha=0.8)
)

plt.tight_layout()
plt.savefig(f"{seed}_both.png", dpi=300)
plt.close(fig)

# ======================
# 2️⃣ UP ONLY
# ======================
fig, ax = plt.subplots(figsize=(fig_w, fig_h))

bs.plot_bs(
    ax,
    spin_index=0,        # ✅ IMPORTANT FIX
    Elim=(ymin, ymax),
    mark_gap=False       # avoid internal bug
)

ax.set_title("Spin Up")

plt.tight_layout()
plt.savefig(f"{seed}_Up.png", dpi=300)
plt.close(fig)

# ======================
# 3️⃣ DOWN ONLY
# ======================
fig, ax = plt.subplots(figsize=(fig_w, fig_h))

bs.plot_bs(
    ax,
    spin_index=1,        # ✅ IMPORTANT FIX
    Elim=(ymin, ymax),
    mark_gap=False
)

ax.set_title("Spin Down")

plt.tight_layout()
plt.savefig(f"{seed}_Down.png", dpi=300)
plt.close(fig)

print("Saved plots:")
print(f"{seed}_both.png")
print(f"{seed}_Up.png")
print(f"{seed}_Down.png")
