# -*- coding: utf-8 -*-
import importlib
import numpy as np
from pymatgen.core import Lattice, Structure, Element


def diamond_si_conventional(a=5.50):
    """Diamond Si conventional cubic cell (8 atoms). (001) aligns with cubic z."""
    lattice = Lattice.cubic(a)
    frac = [
        [0, 0, 0],
        [0, 0.5, 0.5],
        [0.5, 0, 0.5],
        [0.5, 0.5, 0],
        [0.25, 0.25, 0.25],
        [0.25, 0.75, 0.75],
        [0.75, 0.25, 0.75],
        [0.75, 0.75, 0.25],
    ]
    return Structure(lattice, ["Si"] * 8, frac)


def build_sige_si_001(
    a0=5.50,
    nx=2,
    ny=2,
    n_sige=4,     # SiGe segment thickness in conventional-cells along z
    n_si=4,       # Si segment thickness in conventional-cells along z
    x_ge=0.30,    # Ge fraction in SiGe region
    seed=1
):
    """
    Build periodic SiGe/Si superlattice along (001).
    Due to PBC: .../SiGe/Si/SiGe/Si/...

    Steps:
      1) Build pure-Si supercell (nx, ny, nz=n_sige+n_si)
      2) Identify sites with z_frac < n_sige/nz as SiGe region
      3) Randomly replace x_ge fraction of those sites with Ge
    """
    if n_sige <= 0 or n_si <= 0:
        raise ValueError("n_sige and n_si must both be > 0 (so you really have SiGe/Si).")
    if not (0.0 <= x_ge <= 1.0):
        raise ValueError("x_ge must be in [0, 1].")

    nz = int(n_sige + n_si)
    rng = np.random.default_rng(seed)

    st = diamond_si_conventional(a=a0)
    st.make_supercell([int(nx), int(ny), nz])

    # Select SiGe region by fractional z
    z_frac = np.array([site.frac_coords[2] for site in st.sites], dtype=float)
    z_cut = float(n_sige) / float(nz)
    sige_idx = np.where(z_frac < z_cut)[0]

    n_sites = int(len(sige_idx))
    n_ge = int(round(float(x_ge) * n_sites))
    if n_ge > 0:
        pick = rng.choice(sige_idx, size=n_ge, replace=False)
        for idx in pick:
            st[int(idx)] = Element("Ge")  # IMPORTANT: int(idx) for pymatgen compatibility

    return st


def write_atom_config(structure, filename="atom.config", sort_structure=True):
    """
    Write PWmat atom.config via pymatgen's AtomConfig.
    Use dynamic import to avoid PyCharm 'unresolved reference' warnings.
    """
    mod = importlib.import_module("pymatgen.io.pwmat")
    AtomConfig = getattr(mod, "AtomConfig")
    AtomConfig(structure, sort_structure=sort_structure).write_file(filename)


if __name__ == "__main__":
    st = build_sige_si_001(
        a0=5.50,
        nx=2,
        ny=2,
        n_sige=4,
        n_si=4,
        x_ge=0.30,
        seed=1
    )

    # Quick sanity checks (optional, but helpful)
    comp = st.composition
    print("Composition:", comp)
    print("Total atoms:", len(st))

    write_atom_config(st, filename="atom.config", sort_structure=True)
    print("Wrote atom.config")
