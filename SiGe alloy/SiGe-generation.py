#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
make_SiGe_random_alloy.py

功能：
  从 Si 216 原子结构文件（Si 原子 Z=14）中，
  随机把一部分 Si 替换成 Ge (Z=32)，生成给定 Ge 含量的 SiGe 随机合金。

假设输入文件格式类似：
    216
    Lattice vector
      ax ay az
      bx by bz
      cx cy cz
    Position, move_x, move_y, move_z
      14   fx fy fz   1 1 1
      14   ...
      ...

用法示例（在 PyCharm 的 “Parameters” 里）：
    --input atom-Si216.config --output atom-SiGe_x0.30.config --ge_frac 0.30 --seed 123

参数说明：
    --input    输入 config 文件名
    --output   输出 config 文件名
    --ge_frac  Ge 原子占比（0~1 之间），例如 0.30 表示 30 at.%
    --seed     随机种子（可选，指定后结果可复现）
"""

import argparse
import random

# 原子序数
SI_Z = 14
GE_Z = 32


def parse_args():
    parser = argparse.ArgumentParser(description="从 Si216 结构生成随机 SiGe 合金")
    parser.add_argument(
        "--input", "-i", required=True,
        help="输入 Si216 结构文件（例如 atom-Si216.config）"
    )
    parser.add_argument(
        "--output", "-o", required=True,
        help="输出 SiGe 合金结构文件名"
    )
    parser.add_argument(
        "--ge_frac", "-x", type=float, required=True,
        help="Ge 原子摩尔分数（0~1），例如 0.30 表示 30%% Ge"
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="随机种子（可选，指定后结果可复现）"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if not (0.0 <= args.ge_frac <= 1.0):
        raise ValueError("ge_frac 必须在 [0,1] 之间")

    if args.seed is not None:
        random.seed(args.seed)

    # 读入全部行
    with open(args.input, "r", encoding="utf-8", errors="replace") as f:
        lines = f.readlines()

    # 第 1 行：原子数
    try:
        n_atoms = int(lines[0].split()[0])
    except Exception as e:
        raise RuntimeError("无法从第一行解析原子数") from e

    # 找到 “Position, move_x, move_y, move_z” 这一行
    pos_header_idx = None
    for i, line in enumerate(lines):
        if line.strip().lower().startswith("position"):
            pos_header_idx = i
            break

    if pos_header_idx is None:
        raise RuntimeError("未找到 'Position, move_x, move_y, move_z' 行")

    pos_start = pos_header_idx + 1
    pos_end = pos_start + n_atoms
    if pos_end > len(lines):
        raise RuntimeError("文件中原子行数量不够，检查原子数是否正确")

    atom_lines = lines[pos_start:pos_end]

    # 需要替换的 Ge 原子数（四舍五入）
    n_ge = int(round(n_atoms * args.ge_frac))
    if n_ge < 0 or n_ge > n_atoms:
        raise RuntimeError("计算得到的 Ge 原子数不合理: n_ge = {}".format(n_ge))

    # 随机选择哪些原子变成 Ge
    ge_indices = set(random.sample(range(n_atoms), n_ge))

    # 修改原子行
    new_atom_lines = []
    for idx, line in enumerate(atom_lines):
        # 按空格拆分
        parts = line.split()
        if len(parts) < 7:
            # 如果这一行格式异常，就直接保留原行
            new_atom_lines.append(line)
            continue

        # 修改 Z：在 ge_indices 里的设为 Ge，否则设为 Si
        if idx in ge_indices:
            parts[0] = str(GE_Z)
        else:
            parts[0] = str(SI_Z)

        # 重新格式化输出（保持为 7 列：Z, x, y, z, move_x, move_y, move_z）
        # 也可以简单用 " ".join(parts)，这里稍微对齐一下字段。
        try:
            z, x, y, zc, mx, my, mz = parts[:7]
            new_line = f"{int(z):4d} {float(x):14.12f} {float(y):14.12f} {float(zc):14.12f} {int(mx):3d} {int(my):3d} {int(mz):3d}\n"
        except Exception:
            # 如果强制格式化失败，就退回到简单拼接
            new_line = " ".join(parts) + "\n"

        new_atom_lines.append(new_line)

    # 写出新文件：头 + 晶格 + Position 行 + 新的原子行 + 其余行（如果有）
    new_lines = []
    new_lines.extend(lines[:pos_start])      # 前面的内容保持不变
    new_lines.extend(new_atom_lines)         # 替换后的原子行
    new_lines.extend(lines[pos_end:])        # 后面如果还有内容也保留

    with open(args.output, "w", encoding="utf-8") as f:
        f.writelines(new_lines)

    # 在终端打印一下统计信息
    print(f"输入文件: {args.input}")
    print(f"输出文件: {args.output}")
    print(f"总原子数: {n_atoms}")
    print(f"目标 Ge 含量: {args.ge_frac:.4f}")
    print(f"实际 Ge 原子数: {n_ge} (占比 {n_ge / n_atoms:.4f})")


if __name__ == "__main__":
    main()
