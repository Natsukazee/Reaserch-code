#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
atom_config_flagger.py
用法:
  python atom_config_flagger.py -i atom.config -o atom.config.updated -n 19,38

说明:
  - 自动识别以 'position' 开头的区块（不区分大小写）
  - 区块内每行格式: Z  fx fy fz  m1 m2 m3  [flag可选]
    Z为原子序数, fx/fy/fz为分数坐标, m1~m3为是否可动的0/1或整数, 最后flag为你要设定的列
  - 对每个区块独立编号(1-based). 指定编号集合内的原子末尾设为1, 其他设为0
"""

import argparse
import re
from pathlib import Path

ATOM_RE = re.compile(
    r"""
    ^\s*
    (?P<Z>\d+)\s+                              # 原子序数
    (?P<fx>[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)\s+
    (?P<fy>[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)\s+
    (?P<fz>[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)\s+
    (?P<m1>[+-]?\d+)\s+(?P<m2>[+-]?\d+)\s+(?P<m3>[+-]?\d+)
    (?:\s+(?P<flag>[+-]?\d+))?                 # 可选的最后一列
    \s*$
    """,
    re.VERBOSE,
)

def parse_indices(s: str) -> set[int]:
    s = s.strip()
    if not s:
        return set()
    out = set()
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            a, b = int(a), int(b)
            lo, hi = (a, b) if a <= b else (b, a)
            out.update(range(lo, hi + 1))
        else:
            out.add(int(part))
    return out

def process(lines: list[str], target_indices: set[int]) -> tuple[list[str], dict[int,int]]:
    out_lines = []
    in_pos_block = False
    atom_idx = 0
    changed = {}  # block-wise last atom count for info

    for i, line in enumerate(lines):
        stripped = line.strip()
        # 进入新position区块
        if stripped.lower().startswith("position"):
            if in_pos_block:
                # 上一个区块结束
                changed[len(changed)+1] = atom_idx
            in_pos_block = True
            atom_idx = 0
            out_lines.append(line)
            continue

        if in_pos_block:
            m = ATOM_RE.match(line)
            if m:
                atom_idx += 1
                new_flag = 1 if atom_idx in target_indices else 0

                Z = m.group("Z")
                fx, fy, fz = m.group("fx"), m.group("fy"), m.group("fz")
                m1, m2, m3 = m.group("m1"), m.group("m2"), m.group("m3")

                new_line = f"{Z} {fx} {fy} {fz} {m1} {m2} {m3} {new_flag}"
                out_lines.append(new_line)
                continue
            else:
                # 非原子行 -> 认为区块结束
                in_pos_block = False
                changed[len(changed)+1] = atom_idx
                out_lines.append(line)
                continue
        else:
            out_lines.append(line)

    # 文件以position区块结尾的情况
    if in_pos_block:
        changed[len(changed)+1] = atom_idx

    return out_lines, changed

def main():
    ap = argparse.ArgumentParser(description="在 atom.config 的 position 区块为指定原子行末尾添加/设置标志位。")
    ap.add_argument("-i", "--input", required=True, help="输入文件 (atom.config)")
    ap.add_argument("-o", "--output", required=True, help="输出文件 (atom.config.updated)")
    ap.add_argument("-n", "--indices", required=True,
                    help="需要标记为1的原子序号（1-based），逗号分隔，可用区间如 5-10，例如: 19,38 或 1-4,7,9")
    args = ap.parse_args()

    idx_set = parse_indices(args.indices)
    if not idx_set:
        raise SystemExit("错误：未提供有效的原子序号。示例：-n 19,38 或 -n 1-5,9")

    inp = Path(args.input)
    text = inp.read_text(encoding="utf-8")
    lines = text.splitlines()

    new_lines, stats = process(lines, idx_set)

    outp = Path(args.output)
    outp.write_text("\n".join(new_lines) + "\n", encoding="utf-8")

    # 简短报告
    print(f"已写出: {outp}")
    if stats:
        for block_no, count in stats.items():
            print(f"[position 区块 {block_no}] 处理原子数: {count}；标记为1的索引(若在范围内): {sorted([i for i in idx_set if 1 <= i <= count])}")

if __name__ == "__main__":
    main()
