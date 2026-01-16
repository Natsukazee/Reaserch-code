# -*- coding: utf-8 -*-
"""
Windows 一键读取 & 抽帧 & 写出 POSCAR 的脚本
=================================================
使用方法（把脚本与 “样例.txt” 和 “POSCAR_ref” 放在同一文件夹）：
1) 全量写 POSCAR：
   python movement2poscar_windows.py
2) 每隔 N 帧取一帧：
   python movement2poscar_windows.py --frame-step 5
3) 切片范围取帧（start:stop:step，可留空任意一项）：
   python movement2poscar_windows.py --frame-range :20:2
   python movement2poscar_windows.py --frame-range 2:20:3
   python movement2poscar_windows.py --frame-range 5::
4) 指定帧索引列表（0 基）：
   python movement2poscar_windows.py --frames 0,7,9,15
5) 同时指定时优先级： --frames > --frame-range > --frame-step

依赖：pip install numpy pymatgen
输出：在当前目录创建 POSCARS/，生成 POSCAR_0000、POSCAR_0001、…
"""
import os
import re
import sys
import argparse
from typing import List, Tuple
import numpy as np

# -- 可选依赖：仅在写 POSCAR 时才需要 --
try:
    from pymatgen.core import Structure
    from pymatgen.io.vasp import Poscar
    _HAS_PYMATGEN = True
except Exception:
    _HAS_PYMATGEN = False


# ----------------------- I/O & 解析 -----------------------
def _read_text_with_encodings(path: str) -> str:
    """尝试多种常见编码读取文本。"""
    last_err = None
    for enc in ("utf-8", "utf-8-sig", "gb18030", "cp936", "latin1"):
        try:
            with open(path, "r", encoding=enc, errors="strict") as f:
                return f.read()
        except Exception as e:
            last_err = e
    raise RuntimeError(f"无法用常见编码读取文件：{path}\n最后错误：{last_err}")


def read_fractional_frames(filepath: str) -> List[np.ndarray]:
    """
    读取 '样例.txt' 里的每一帧分数坐标，仅提取 Position 段第 2~4 列为 (fx, fy, fz).
    返回：frames (list[np.ndarray])，每帧形状 (natoms, 3).
    """
    txt = _read_text_with_encodings(filepath)

    # 统一换行，兼容 Windows \r\n
    txt = txt.replace("\r\n", "\n").replace("\r", "\n")

    # 用由多条 '-' 组成的整行作为帧分隔
    blocks = re.split(r'\n\s*-{5,}\s*\n', txt)

    frames: List[np.ndarray] = []
    natoms_hint = None

    for blk in blocks:
        if not blk.strip():
            continue

        # 可选：从帧头推断原子数（不强制）
        m_na = re.search(r'^\s*(\d+)\s+atoms', blk, flags=re.M)
        if m_na:
            try:
                natoms_hint = int(m_na.group(1))
            except Exception:
                natoms_hint = None

        # 截取 Position 段（到 -Force 或 Velocity 之前）
        m = re.search(r'Position.*?\n(.*?)(?:\n\s*-Force|\nVelocity)', blk, flags=re.S)
        if not m:
            # 更宽松的兜底：遇到新标题、分隔线或文本结束则截断
            m = re.search(r'Position.*?\n(.*?)(?:\n[A-Za-z].*?:|\n\s*-{3,}|\Z)', blk, flags=re.S)
            if not m:
                continue

        pos_section = m.group(1)
        coords = []
        for line in pos_section.splitlines():
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            # 期望：ID fx fy fz [1 1 1]
            if len(parts) < 4:
                continue
            try:
                x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
                coords.append([x, y, z])
            except Exception:
                continue

        if coords:
            arr = np.array(coords, dtype=float)
            # 如果能拿到 natoms_hint，则做个友好校验但不强制
            if natoms_hint is not None and arr.shape[0] != natoms_hint:
                print(f"[warn] 本帧读取到 {arr.shape[0]} 个原子，与头部 {natoms_hint} 不一致。", file=sys.stderr)
            frames.append(arr)

    if not frames:
        raise ValueError("没有解析到任何帧。请检查 '样例.txt' 是否包含 Position 段。")

    # 校验每帧原子数一致（严格）
    nset = {arr.shape[0] for arr in frames}
    if len(nset) != 1:
        raise ValueError(f"不同帧原子数不一致：{sorted(nset)}")

    print(f"✅ 成功读取 {len(frames)} 帧；每帧 {frames[0].shape[0]} 个原子。")
    return frames


# ----------------------- 抽帧逻辑 -----------------------
def select_frames(frames: List[np.ndarray],
                  frames_arg: str = None,
                  range_arg: str = None,
                  step_arg: int = 1) -> Tuple[List[np.ndarray], List[int]]:
    """根据用户抽帧参数选择子集并返回 (selected_frames, indices)。优先级：frames > range > step。"""
    n = len(frames)
    idx: List[int] = None

    if frames_arg:  # 最高优先级: 显式索引列表
        idx = [int(x) for x in frames_arg.split(',') if x.strip() != '']

    elif range_arg:  # 次优先级: 切片风格
        parts = range_arg.split(':')
        if len(parts) > 3:
            raise ValueError("--frame-range 格式应为 start:stop:step（如 2:20:3 或 :100:2 或 5::）")
        def _to_int_or_none(s: str):
            s = s.strip()
            return int(s) if s != '' else None
        start = _to_int_or_none(parts[0]) if len(parts) >= 1 else None
        stop  = _to_int_or_none(parts[1]) if len(parts) >= 2 else None
        step  = _to_int_or_none(parts[2]) if len(parts) >= 3 else None
        rng = range(n)[slice(start, stop, step)]
        idx = list(rng)

    elif step_arg and step_arg > 1:  # 基础步长抽帧
        idx = list(range(0, n, step_arg))

    if idx is None:  # 未指定抽帧 => 全部
        idx = list(range(n))

    # 合法化与去重
    idx = sorted(set(i for i in idx if 0 <= i < n))
    sel = [frames[i] for i in idx]
    if not sel:
        raise ValueError("抽帧结果为空，请检查索引/范围是否超出。")
    return sel, idx


# ----------------------- 写 POSCAR -----------------------
def write_poscars(frames: List[np.ndarray], template_path: str, outdir: str) -> None:
    """将每一帧写成 POSCAR_* 文件。"""
    if not _HAS_PYMATGEN:
        raise RuntimeError("写 POSCAR 需要 pymatgen，请先安装：pip install pymatgen")

    if not os.path.exists(template_path):
        raise FileNotFoundError(f"未找到模板：{template_path}")

    struct0 = Structure.from_file(template_path)
    natoms = len(struct0)

    os.makedirs(outdir, exist_ok=True)

    for i, frac in enumerate(frames):
        if frac.shape[0] != natoms:
            raise ValueError(f"第 {i} 帧原子数 {frac.shape[0]} 与模板 {natoms} 不一致。")

        s = struct0.copy()
        for j in range(natoms):
            # 分数坐标替换
            s.replace(j, species=s[j].species, coords=frac[j], coords_are_cartesian=False)

        outp = os.path.join(outdir, f"POSCAR_{i:04d}")
        Poscar(s).write_file(outp)

    print(f"📦 已输出 {len(frames)} 个 POSCAR 到：{outdir}")


# ----------------------- 主程序 -----------------------
def main():
    parser = argparse.ArgumentParser(
        description="读取 样例.txt，支持抽帧，并基于 POSCAR_ref 写出 POSCAR_* 文件到 POSCARS/")
    parser.add_argument("--frame-step", type=int, default=1,
                        help="每隔 N 帧取一帧（默认 1，不抽帧）。")
    parser.add_argument("--frame-range", type=str, default=None,
                        help="像切片那样选择帧：start:stop:step，例如 2:20:3 或 :100:2 或 5::")
    parser.add_argument("--frames", type=str, default=None,
                        help="精确帧索引的逗号列表，例如 0,7,9,15（0 基索引）。")
    parser.add_argument("--save-npy", type=str, default=None,
                        help="可选：将抽帧后的分数坐标保存为 .npy（形状 n_frames × natoms × 3）。")
    args = parser.parse_args()

    base = os.getcwd()
    input_file = os.path.join(base, "MOVEMENT.txt")
    template_file = os.path.join(base, "POSCAR_ref")
    outdir = os.path.join(base, "POSCARS")

    if not os.path.exists(input_file):
        print("❌ 未找到 '样例.txt'。请把本脚本放到含有 '样例.txt' 的文件夹中运行。")
        sys.exit(1)
    if not os.path.exists(template_file):
        print("❌ 未找到 'POSCAR_ref'。请把模板 'POSCAR_ref' 放在同一文件夹。")
        sys.exit(1)

    try:
        frames_all = read_fractional_frames(input_file)
        frames_sel, picked = select_frames(frames_all, args.frames, args.frame_range, args.frame_step)
        print(f"🧮 抽帧后保留 {len(frames_sel)} 帧，索引：{picked}")

        if args.save_npy:
            stacked = np.stack(frames_sel, axis=0)
            np.save(args.save_npy, stacked)
            print(f"💾 已保存分数坐标到：{args.save_npy}（形状 {stacked.shape}）")

        write_poscars(frames_sel, template_file, outdir)
        print("🎉 完成！")

    except Exception as e:
        print("程序出错：", e)
        sys.exit(2)


if __name__ == "__main__":
    main()
