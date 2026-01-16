#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Keating VFF strain energy (robust parser, no need for 'Position' header).
- Auto-detect lattice (first 3 lines with >=3 floats) -> orthorhombic only
- Auto-detect atom block (lines: Z fx fy fz [flags...]) without 'Position'
- Builds bonds via cutoffs; computes stretch/bend energies
- Toggle beta with --include-beta; export per-bond/per-angle CSV

params.json 示例：
{
  "alpha": { "C-C": 100.0, "C-Si": 80.0, "Si-Si": 50.0 },
  "beta":  { "C-Si-C": 15.0, "Si-C-Si": 12.0, "*-Si-*": 10.0 },
  "d0":    { "C-C": 1.54,   "C-Si": 1.89,  "Si-Si": 2.35 },
  "cutoff":{ "C-C": 1.8,    "C-Si": 2.2,   "Si-Si": 2.8  }
}
"""

import argparse, json, math, csv, re
from collections import defaultdict
import numpy as np

Z2SYM = {
    1:"H", 6:"C", 14:"Si", 32:"Ge", 50:"Sn",
    # 如需其它元素，自行补充或统一用 Z{num}
}

# -------------------- 小工具 --------------------
_float_re = re.compile(r'^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?$')

def is_float(tok: str) -> bool:
    return bool(_float_re.match(tok))

def try_float(tok: str):
    return float(tok) if is_float(tok) else None

def is_int(tok: str) -> bool:
    try:
        int(tok); return True
    except: return False

# -------------------- 解析器（无需 Position） --------------------
def read_atom_config(path):
    """
    读取 atom.config：自动取第一组 3 行(每行>=3个浮点)为晶格向量；
    随后自动识别原子块：行首是 Z (1..118)，后接 ≥3 浮点 (fx fy fz)。
    遇到非原子行/Force/Stress 结束。
    返回：species(符号), frac(N×3), lat(3×3)
    """
    with open(path, "r", encoding="utf-8") as f:
        # 过滤纯空行；保留其它行（注释行也保留，后面跳过）
        raw = [ln.rstrip() for ln in f if ln.strip()]

    # 1) 抓晶格：找第一段“连续三行，每行>=3个浮点”的块
    lat_rows = []
    i = 0
    while i < len(raw) and not lat_rows:
        # 尝试用 i, i+1, i+2
        if i+2 < len(raw):
            ok = True
            triple = []
            for k in (i, i+1, i+2):
                toks = raw[k].split()
                floats = [try_float(t) for t in toks if is_float(t)]
                if len(floats) < 3:
                    ok = False; break
                triple.append(floats[:3])
            if ok:
                lat_rows = triple
                i = i+3
                break
        i += 1
    if not lat_rows:
        raise ValueError("未能在文件前部找到 3×3 晶格向量（每行≥3个浮点）。")
    lat = np.array(lat_rows, dtype=float)

    # 2) 抓原子：从 lattice 块后继续往下搜
    species = []
    frac = []
    j = i
    while j < len(raw):
        ln = raw[j]
        low = ln.lower()
        if low.startswith("force") or low.startswith("stress"):
            break
        toks = ln.split()
        if not toks:
            j += 1; continue

        # 允许首列为 Z；也容忍“index Z fx fy fz ...”的2列整数前缀格式
        # 先试标准：Z fx fy fz
        parse_ok = False
        if is_int(toks[0]) and len(toks) >= 4 and all(is_float(x) for x in toks[1:4]):
            Z = int(toks[0]); fx, fy, fz = map(float, toks[1:4])
            parse_ok = True
            rest_consumed = 4
        elif len(toks) >= 5 and is_int(toks[0]) and is_int(toks[1]) and all(is_float(x) for x in toks[2:5]):
            # index Z fx fy fz
            Z = int(toks[1]); fx, fy, fz = map(float, toks[2:5])
            parse_ok = True
            rest_consumed = 5

        if parse_ok and (1 <= Z <= 118):
            species.append(Z2SYM.get(Z, f"Z{Z}"))
            frac.append([fx, fy, fz])
            j += 1
            continue

        # 命中非原子行 => 如果已经开始积累过原子，说明原子块结束；否则跳过此行继续找
        if species:
            break
        j += 1

    if not species:
        raise ValueError("未识别到任何原子行。请确认行首是原子序号、后跟3个坐标。")

    frac = np.array(frac, dtype=float)
    species = np.array(species, dtype=object)
    return species, frac, lat

# -------------------- PBC & 几何 --------------------
def is_orthorhombic(lat, tol=1e-8):
    offdiag = np.array([lat[0,1], lat[0,2], lat[1,0], lat[1,2], lat[2,0], lat[2,1]])
    return np.all(np.abs(offdiag) < tol)

def frac_to_cart(frac, lat):  # 行向量 × 矩阵
    return frac @ lat

def min_image_vec(dr, box_diag):
    out = dr.copy()
    for d in range(3):
        L = box_diag[d]
        out[d] -= round(out[d]/L) * L
    return out

def dist_orthorhombic(rj_ri, box_diag):
    return float(np.linalg.norm(min_image_vec(rj_ri, box_diag)))

# -------------------- 拓扑 --------------------
def pair_key(a, b): return "-".join(sorted((a, b)))
def triplet_key(j, i, k): return f"{j}-{i}-{k}"

def get_pair_cutoff(a, b, cutoff_map, global_cut):
    key = pair_key(a, b)
    if cutoff_map and key in cutoff_map:
        return cutoff_map[key]
    return global_cut

def build_bonds(species, cart, cutoff_map, global_cut, box_diag):
    N = len(species)
    bonds, by_center = [], defaultdict(list)
    for i in range(N):
        for j in range(i+1, N):
            cut = get_pair_cutoff(species[i], species[j], cutoff_map, global_cut)
            if cut is None: continue
            d = dist_orthorhombic(cart[j]-cart[i], box_diag)
            if d <= cut:
                bonds.append((i,j))
                by_center[i].append(j); by_center[j].append(i)
    return bonds, by_center

def build_angles(by_center):
    angles = []
    for i, neighs in by_center.items():
        if len(neighs) < 2: continue
        Ns = sorted(neighs)
        for a in range(len(Ns)):
            for b in range(a+1, len(Ns)):
                j, k = Ns[a], Ns[b]
                angles.append((j,i,k))
    return angles

# -------------------- Keating VFF --------------------
def stretch_energy(species, cart, bonds, alpha_map, d0_map, box_diag, writer=None):
    U = 0.0
    for (i, j) in bonds:
        key = pair_key(species[i], species[j])
        if key not in alpha_map or key not in d0_map:
            raise KeyError(f"缺少 alpha/d0：'{key}'")
        alpha = alpha_map[key]; d0 = d0_map[key]
        rij = min_image_vec(cart[j]-cart[i], box_diag)
        r2 = float(np.dot(rij, rij))
        term = (r2 - d0**2)
        U_ij = (3.0 * alpha / (16.0 * d0**2)) * term**2
        U += U_ij
        if writer:
            writer.writerow({
                "i": i, "j": j, "species_i": species[i], "species_j": species[j],
                "r(Å)": math.sqrt(r2), "d0(Å)": d0, "alpha(eV/A^2)": alpha,
                "U_stretch_ij(eV)": U_ij
            })
    return U

def bend_energy(species, cart, angles, beta_map, d0_map, box_diag, writer=None):
    U = 0.0
    for (j, i, k) in angles:
        key_ij = pair_key(species[i], species[j])
        key_ik = pair_key(species[i], species[k])
        if key_ij not in d0_map or key_ik not in d0_map:
            raise KeyError(f"缺少 d0：'{key_ij}' 或 '{key_ik}'")
        d0ij, d0ik = d0_map[key_ij], d0_map[key_ik]

        tkey = triplet_key(species[j], species[i], species[k])
        beta = beta_map.get(tkey, None)
        if beta is None:
            tkey_sym = triplet_key(species[k], species[i], species[j])
            beta = beta_map.get(tkey_sym, None)
        if beta is None:
            center_fallback = f"*-{species[i]}-*"
            beta = beta_map.get(center_fallback, 0.0)

        rij = min_image_vec(cart[j]-cart[i], box_diag)
        rik = min_image_vec(cart[k]-cart[i], box_diag)
        dotp = float(np.dot(rij, rik))
        term = dotp + (d0ij * d0ik)/3.0
        pref = (3.0 * beta) / (8.0 * d0ij * d0ik)
        U_jik = pref * term**2
        U += U_jik

        if writer:
            cos_theta = dotp / (np.linalg.norm(rij)*np.linalg.norm(rik) + 1e-15)
            cos_theta = max(min(cos_theta, 1.0), -1.0)
            theta = math.degrees(math.acos(cos_theta))
            writer.writerow({
                "j": j, "i": i, "k": k,
                "species_j": species[j], "species_i": species[i], "species_k": species[k],
                "theta(deg)": theta, "beta(eV/A^2)": beta,
                "d0_ij(Å)": d0ij, "d0_ik(Å)": d0ik, "U_bend_jik(eV)": U_jik
            })
    return U

# -------------------- 主程序 --------------------
def main():
    ap = argparse.ArgumentParser(description="Keating VFF strain energy (no 'Position' needed)")
    ap.add_argument("--config", required=True, help="atom.config 路径")
    ap.add_argument("--params", required=True, help="参数 JSON：alpha/beta/d0/(cutoff)")
    ap.add_argument("--pbc", action="store_true", help="启用正交 PBC（晶格从文件自动读取）")
    ap.add_argument("--global-cutoff", type=float, default=None, help="默认建键阈值(Å)")
    ap.add_argument("--include-beta", dest="include_beta", action="store_true",
                    help="包含键角能（β≠0）；不加则仅伸缩能")
    ap.add_argument("--per-bond", help="导出逐键伸缩能 CSV")
    ap.add_argument("--per-angle", help="导出逐角弯曲能 CSV")
    args = ap.parse_args()

    species, frac, lat = read_atom_config(args.config)

    if args.pbc:
        if not is_orthorhombic(lat):
            raise NotImplementedError("当前仅支持正交晶胞。如需非正交，请告知我升级 3×3 版本。")
        box_diag = np.array([lat[0,0], lat[1,1], lat[2,2]], dtype=float)
        print(f"Detected orthorhombic box: {tuple(box_diag)}")
    else:
        # 近似“无边界”——不做最小像折返
        box_diag = np.array([1e12, 1e12, 1e12], dtype=float)

    cart = frac_to_cart(frac, lat)

    with open(args.params, "r", encoding="utf-8") as f:
        prm = json.load(f)
    alpha_map = prm.get("alpha", {})
    beta_map  = prm.get("beta",  {})
    d0_map    = prm.get("d0",    {})
    cutoff_map= prm.get("cutoff",{})

    bonds, by_center = build_bonds(species, cart, cutoff_map, args.global_cutoff, box_diag)
    angles = build_angles(by_center)

    # CSV writers
    bond_writer = angle_writer = None
    bf = af = None
    if args.per_bond:
        bf = open(args.per_bond, "w", newline="", encoding="utf-8")
        bond_writer = csv.DictWriter(bf, fieldnames=[
            "i","j","species_i","species_j","r(Å)","d0(Å)","alpha(eV/A^2)","U_stretch_ij(eV)"
        ])
        bond_writer.writeheader()
    if args.per_angle:
        af = open(args.per_angle, "w", newline="", encoding="utf-8")
        angle_writer = csv.DictWriter(af, fieldnames=[
            "j","i","k","species_j","species_i","species_k",
            "theta(deg)","beta(eV/A^2)","d0_ij(Å)","d0_ik(Å)","U_bend_jik(eV)"
        ])
        angle_writer.writeheader()

    try:
        U_s = stretch_energy(species, cart, bonds, alpha_map, d0_map, box_diag, writer=bond_writer)
        U_b = 0.0
        if args.include_beta:
            U_b = bend_energy(species, cart, angles, beta_map, d0_map, box_diag, writer=angle_writer)
    finally:
        if bf: bf.close()
        if af: af.close()

    print(f"# Atoms: {len(species)}   Bonds: {len(bonds)}   Angles: {len(angles)}")
    print(f"U_stretch (eV) = {U_s:.6f}")
    print(f"U_bend   (eV) = {U_b:.6f}   (beta {'ON' if args.include_beta else 'OFF'})")
    print(f"U_total  (eV) = {U_s + U_b:.6f}")

if __name__ == "__main__":
    main()
