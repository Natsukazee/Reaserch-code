import re
import math
from pathlib import Path
import numpy as np


# ============================================================
# Read PWmat-style .config (Force section optional)
# ============================================================

def read_pwmat_config(path: str) -> dict:
    lines = Path(path).read_text().splitlines()

    # ---------- natoms ----------
    m = re.search(r"(\d+)\s+atoms", lines[0], re.IGNORECASE)
    if not m:
        raise ValueError(f"Cannot parse natoms from header line in {path}")
    natoms = int(m.group(1))

    # ---------- LATTICE ----------
    try:
        lat_hdr = next(i for i, l in enumerate(lines) if l.strip().upper() == "LATTICE")
    except StopIteration:
        raise ValueError(f"No LATTICE section found in {path}")

    lattice = np.array(
        [[float(x) for x in lines[lat_hdr + i].split()[:3]] for i in range(1, 4)],
        dtype=float
    )

    # ---------- POSITION ----------
    try:
        pos_hdr = next(i for i, l in enumerate(lines) if l.strip().upper().startswith("POSITION"))
    except StopIteration:
        raise ValueError(f"No POSITION section found in {path}")

    pos_start = pos_hdr + 1
    pos_lines = lines[pos_start:pos_start + natoms]
    if len(pos_lines) != natoms:
        raise ValueError(f"Position lines count mismatch in {path}")

    pos = []
    for l in pos_lines:
        toks = l.split()
        z = int(toks[0])
        x, y, zf = map(float, toks[1:4])
        moves = list(map(int, toks[4:7])) if len(toks) >= 7 else [1, 1, 1]
        pos.append((z, x, y, zf, moves[0], moves[1], moves[2]))

    # ---------- FORCE (optional) ----------
    force_line_idx = None
    for i, l in enumerate(lines):
        if l.strip().upper() == "FORCE":
            force_line_idx = i
            break

    if force_line_idx is None:
        # No Force → auto-fill zero forces
        forces = [(z, 0.0, 0.0, 0.0) for z, *_ in pos]
        force_line_raw = " FORCE"
        tail_lines = []
    else:
        force_start = force_line_idx + 1
        force_lines = lines[force_start:force_start + natoms]
        if len(force_lines) != natoms:
            raise ValueError(f"Force lines count mismatch in {path}")

        forces = []
        for l in force_lines:
            toks = l.split()
            z = int(toks[0])
            fx, fy, fz = map(float, toks[1:4])
            forces.append((z, fx, fy, fz))

        force_line_raw = lines[force_line_idx]
        tail_lines = lines[force_start + natoms:]

    header_lines = lines[:pos_start]

    return {
        "path": path,
        "natoms": natoms,
        "lattice": lattice,
        "header_lines": header_lines,
        "pos": pos,
        "forces": forces,
        "force_line_raw": force_line_raw,
        "tail_lines": tail_lines,
    }


# ============================================================
# Geometry helpers
# ============================================================

def pbc_cart_dist(frac_a, frac_b, lattice) -> float:
    d = np.array(frac_b) - np.array(frac_a)
    d -= np.round(d)
    cart = d @ lattice
    return float(np.linalg.norm(cart))


def wrap01(x: float) -> float:
    return x - math.floor(x)


# ============================================================
# Atom matching (one special atom allowed)
# ============================================================

def improved_match_one_special(struct_a: dict, struct_b: dict):
    a_pos = struct_a["pos"]
    b_pos = struct_b["pos"]
    lat = struct_a["lattice"]
    nat = struct_a["natoms"]

    nn = []
    for (Za, xa, ya, za, *_m) in a_pos:
        best = 1e9
        for (Zb, xb, yb, zb, *_m2) in b_pos:
            if Za != Zb:
                continue
            d = pbc_cart_dist((xa, ya, za), (xb, yb, zb), lat)
            best = min(best, d)
        nn.append(best)

    nn = np.array(nn)
    med = float(np.median(nn))
    mad = float(np.median(np.abs(nn - med))) or 1e-12
    tol = max(0.10, med + 8.0 * 1.4826 * mad)

    pairs = []
    for i, (Za, xa, ya, za, *_m) in enumerate(a_pos):
        for j, (Zb, xb, yb, zb, *_m2) in enumerate(b_pos):
            if Za != Zb:
                continue
            d = pbc_cart_dist((xa, ya, za), (xb, yb, zb), lat)
            pairs.append((d, i, j))
    pairs.sort()

    close_map = {}
    used_a, used_b = set(), set()
    for _ in range(10):
        close_map.clear()
        used_a.clear()
        used_b.clear()
        for d, i, j in pairs:
            if d > tol:
                break
            if i in used_a or j in used_b:
                continue
            close_map[i] = (j, d)
            used_a.add(i)
            used_b.add(j)
        if len(close_map) >= nat - 1:
            break
        tol *= 1.5

    if len(close_map) < nat - 1:
        raise RuntimeError("Failed to match N-1 atoms robustly.")

    unmatched_a = [i for i in range(nat) if i not in close_map]
    unmatched_b = [j for j in range(nat) if j not in used_b]

    special_a = unmatched_a[0]
    special_b = unmatched_b[0]

    full_map = dict(close_map)
    Za, xa, ya, za, *_ = a_pos[special_a]
    Zb, xb, yb, zb, *_ = b_pos[special_b]
    if Za != Zb:
        raise RuntimeError("Special atom element mismatch.")
    d_special = pbc_cart_dist((xa, ya, za), (xb, yb, zb), lat)
    full_map[special_a] = (special_b, d_special)

    return full_map, special_a, special_b, tol, med, mad


# ============================================================
# Output formatting
# ============================================================

def format_position_line(z, x, y, zc, mx, my, mz) -> str:
    return f"{z:4d}  {x:14.9f}  {y:14.9f}  {zc:14.9f}     {mx:d}  {my:d}  {mz:d}"


def format_force_line(z, fx, fy, fz) -> str:
    return f"{z:4d}  {fx:14.9f}  {fy:14.9f}  {fz:14.9f}"


def write_reordered(struct: dict, order: list[int], outpath: str) -> None:
    pos_new = []
    for idx in order:
        pos_new.append(format_position_line(*struct["pos"][idx]))

    force_new = []
    for idx in order:
        force_new.append(format_force_line(*struct["forces"][idx]))

    lines_out = []
    lines_out.extend(struct["header_lines"])
    lines_out.extend(pos_new)
    lines_out.append(struct["force_line_raw"])
    lines_out.extend(force_new)
    lines_out.extend(struct["tail_lines"])

    Path(outpath).write_text("\n".join(lines_out) + "\n")


# ============================================================
# Main reorder routine
# ============================================================

def reorder_two_configs(path_a: str, path_b: str,
                        out_a: str, out_b: str,
                        special_position: str = "last") -> None:

    sa = read_pwmat_config(path_a)
    sb = read_pwmat_config(path_b)

    mapping, special_a, special_b, tol, med, mad = improved_match_one_special(sa, sb)

    nat = sa["natoms"]
    close_a = [i for i in range(nat) if i != special_a]

    key_list = []
    for i in close_a:
        z, x, y, zc, *_ = sa["pos"][i]
        key_list.append((wrap01(x), wrap01(y), wrap01(zc), z, i))
    key_list.sort()

    close_sorted_a = [t[-1] for t in key_list]
    close_sorted_b = [mapping[i][0] for i in close_sorted_a]

    if special_position == "last":
        order_a = close_sorted_a + [special_a]
        order_b = close_sorted_b + [special_b]
    else:
        raise ValueError("Only 'last' is supported")

    write_reordered(sa, order_a, out_a)
    write_reordered(sb, order_b, out_b)

    print("======================================")
    print("Reorder finished")
    print("Tolerance (Å):", tol)
    print("NN median / MAD (Å):", med, mad)
    print("Special atom A (1-based):", special_a + 1)
    print("Special atom B (1-based):", special_b + 1)
    print("Output:", out_a, out_b)


# ============================================================
# Entry
# ============================================================

if __name__ == "__main__":
    reorder_two_configs(
        path_a="atom_isolated.config",
        path_b="atom-colorcenter.config",
        out_a="atom_isolated_reordered.config",
        out_b="atom-colorcenter_reordered.config",
        special_position="last",
    )
