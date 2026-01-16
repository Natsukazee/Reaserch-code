# -*- coding: utf-8 -*-
"""
Atomic path (merged)
====================
功能：
- 从一个包含连续帧 POSCAR 的文件夹中，读取每帧结构；
- 估计每帧相对第0帧的整体平移（排除缺陷原子影响），将缺陷原子的坐标对齐到参考坐标系；
- **将所有帧的“缺陷原子位置”合并到一份 POSCAR 中**，便于一次性可视化缺陷轨迹；
- 同时保留原脚本逐帧写出（可开关）。

依赖：pip install numpy pymatgen
建议可视化：VESTA / OVITO 等

可配置项见“用户设置”部分；默认会在输出目录下生成：
- POSCAR_all_defects        （合并的缺陷轨迹，默认：仅缺陷原子，不含其它原子）
- translations_and_defect_coords.csv  （每帧的估计平移向量与缺陷校正后坐标）
- （可选）Aligned_POSCARS/PerFrame_POSCARS/POSCAR_frame_xxxx   （逐帧，仅缺陷移动，其它原子与第0帧一致）
"""
import os
import numpy as np
import warnings
from typing import List, Tuple
from pymatgen.core import Structure
from pymatgen.io.vasp import Poscar

warnings.filterwarnings("ignore", message="Ignoring selective dynamics tag")

# ========== 用户设置（按需修改） ==========
input_folder = "./POSCARs"                 # 输入帧目录（按文件名排序）
output_folder = "./Aligned_POSCARS"        # 输出目录
defect_index = 18                          # 缺陷原子索引（从0开始）
method = "mean"                            # 平移估计方法："mean" 或 "median"

write_per_frame_aligned = False            # 是否保留逐帧输出（与原脚本一致逻辑）
write_merged_poscar = True                 # 是否写出合并缺陷轨迹 POSCAR（核心需求）

# 合并 POSCAR 的写法：
merged_mode = "overlay_on_ref"
#   - "defects_only" : 只写所有帧的缺陷原子（不包含其它原子）
#   - "overlay_on_ref" : 在第0帧的基础上叠加所有帧的缺陷原子
remove_original_defect_in_overlay = True   # 当 overlay_on_ref 时，是否移除原始那个缺陷原子（避免重叠）
# ========================================


def _read_sorted_poscars(folder: str) -> List[str]:
    files = sorted([f for f in os.listdir(folder) if f.lower().startswith("poscar")])
    if not files:
        raise FileNotFoundError(f"在 {folder} 中没有找到 POSCAR 文件")
    return [os.path.join(folder, f) for f in files]


def _least_image(delta_frac: np.ndarray) -> np.ndarray:
    """最小像处理，把分数坐标差移到 [-0.5, 0.5] 区间。"""
    return delta_frac - np.round(delta_frac)


def _estimate_translation(delta_cart: np.ndarray, defect_index: int, method: str) -> np.ndarray:
    """排除缺陷原子后，用 mean/median 估计整体平移（在笛卡尔坐标下）。"""
    n_atoms = delta_cart.shape[0]
    mask = np.arange(n_atoms) != defect_index
    data = delta_cart[mask]
    if method == "mean":
        return np.mean(data, axis=0)
    elif method == "median":
        return np.median(data, axis=0)
    else:
        raise ValueError("method must be 'mean' or 'median'")


def _collect_aligned_defect_positions(paths: List[str], defect_index: int, method: str
                                      ) -> Tuple[Structure, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """读取所有帧，返回：
       - ref_struct：第0帧结构
       - lattice：第0帧晶格矩阵(3x3)
       - defect_cart_all：(n_frames, 3) 对齐后的缺陷笛卡尔坐标
       - translations_all：(n_frames, 3) 每帧估计的平移向量
       - ref_cart：(N,3) 参考帧原子笛卡尔坐标
    """
    ref_struct = Structure.from_file(paths[0])
    n_atoms = len(ref_struct)
    if not (0 <= defect_index < n_atoms):
        raise IndexError(f"defect_index={defect_index} 超出原子总数 {n_atoms}")

    ref_frac = np.array(ref_struct.frac_coords)   # (N,3)
    ref_cart = np.array(ref_struct.cart_coords)   # (N,3)
    lattice = ref_struct.lattice.matrix           # (3,3)

    defect_cart_all = []
    translations_all = []

    for i, p in enumerate(paths):
        s = Structure.from_file(p)
        if len(s) != n_atoms:
            raise ValueError(f"文件 {os.path.basename(p)} 原子数与参考不一致：{len(s)} vs {n_atoms}")

        frac = np.array(s.frac_coords)
        delta_frac = _least_image(frac - ref_frac)
        delta_cart = delta_frac @ lattice

        t = _estimate_translation(delta_cart, defect_index, method)
        translations_all.append(t)

        # 原始缺陷坐标（当前帧）
        defect_cart_raw = np.array(s.cart_coords[defect_index])
        # 校正到参考坐标系（减去整体平移）
        defect_cart_adj = defect_cart_raw - t
        defect_cart_all.append(defect_cart_adj)

    return ref_struct, lattice, np.vstack(defect_cart_all), np.vstack(translations_all), ref_cart


def write_per_frame_series(ref_struct: Structure,
                           defect_cart_all: np.ndarray,
                           translations_all: np.ndarray,
                           output_folder: str,
                           defect_index: int) -> None:
    """逐帧写出 POSCAR（除了缺陷原子外，其它原子使用第0帧）"""
    os.makedirs(output_folder, exist_ok=True)
    n_frames = defect_cart_all.shape[0]

    for i in range(n_frames):
        new_struct = ref_struct.copy()
        new_struct.replace(defect_index,
                           species=new_struct[defect_index].species,
                           coords=defect_cart_all[i],
                           coords_are_cartesian=True)
        out_name = f"POSCAR_frame_{i:04d}"
        out_path = os.path.join(output_folder, out_name)
        Poscar(new_struct).write_file(out_path)


def write_merged_poscar_defects_only(ref_struct: Structure,
                                     defect_cart_all: np.ndarray,
                                     out_path: str) -> None:
    """只写所有帧的缺陷原子（不包含其它原子），物种使用缺陷原子本身的物种。"""
    species_list = [ref_struct[defect_index].species] * defect_cart_all.shape[0]
    # 直接构建结构（使用参考晶格）
    from pymatgen.core.structure import Structure as PmgStructure
    lattice = ref_struct.lattice
    merged = PmgStructure(lattice=lattice, species=species_list, coords=defect_cart_all, coords_are_cartesian=True)
    Poscar(merged).write_file(out_path)


def write_merged_poscar_overlay(ref_struct: Structure,
                                defect_cart_all: np.ndarray,
                                out_path: str,
                                remove_original_defect: bool) -> None:
    """在第0帧基础上叠加所有帧缺陷原子坐标。若 remove_original_defect=True，则移除原始那个缺陷位点。"""
    merged = ref_struct.copy()

    # 是否移除原始缺陷原子（避免与第一个轨迹点重叠）
    if remove_original_defect:
        merged.pop(defect_index)

    species_def = ref_struct[defect_index].species
    for pos in defect_cart_all:
        merged.append(species=species_def, coords=pos, coords_are_cartesian=True)

    Poscar(merged).write_file(out_path)


def main():
    # 读取与排序
    paths = _read_sorted_poscars(input_folder)

    # 读取 + 对齐 + 收集缺陷坐标
    ref_struct, lattice, defect_cart_all, translations_all, ref_cart = _collect_aligned_defect_positions(
        paths, defect_index, method
    )

    # 输出目录与 CSV
    os.makedirs(output_folder, exist_ok=True)
    csv_path = os.path.join(output_folder, "translations_and_defect_coords.csv")
    with open(csv_path, "w") as f:
        f.write("frame,tx(A),ty(A),tz(A),def_x(A),def_y(A),def_z(A)\n")
        for i in range(defect_cart_all.shape[0]):
            tx, ty, tz = translations_all[i]
            dx, dy, dz = defect_cart_all[i]
            f.write(f"{i},{tx:.6f},{ty:.6f},{tz:.6f},{dx:.6f},{dy:.6f},{dz:.6f}\n")

    print(f"对齐完成：帧数 = {defect_cart_all.shape[0]}，输出 CSV：{os.path.abspath(csv_path)}")

    # 逐帧输出（可选）
    if write_per_frame_aligned:
        series_dir = os.path.join(output_folder, "PerFrame_POSCARS")
        write_per_frame_series(ref_struct, defect_cart_all, translations_all, series_dir, defect_index)
        print(f"逐帧 POSCAR 写出到：{os.path.abspath(series_dir)}")

    # 合并输出（核心）
    if write_merged_poscar:
        merged_path = os.path.join(output_folder, "POSCAR_all_defects")
        if merged_mode == "defects_only":
            write_merged_poscar_defects_only(ref_struct, defect_cart_all, merged_path)
            print(f"合并 POSCAR（仅缺陷原子）已写出：{os.path.abspath(merged_path)}")
        elif merged_mode == "overlay_on_ref":
            write_merged_poscar_overlay(ref_struct, defect_cart_all, merged_path,
                                        remove_original_defect_in_overlay)
            print(f"合并 POSCAR（叠加到参考结构）已写出：{os.path.abspath(merged_path)}")
        else:
            raise ValueError("merged_mode 只能为 'defects_only' 或 'overlay_on_ref'")

    print("完成。")


if __name__ == "__main__":
    main()
