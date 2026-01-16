#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
vesta_colorize_sitet_tricolor.py

在 .vesta 文件的 SITET 段中，把“最后 N 个原子”的颜色设为
红 -> 白 -> 蓝 的三段渐变（同时修改两处 RGB 三元组，一般对应球和键）。

用法示例：
  python vesta_colorize_sitet_tricolor.py input.vesta -n 20
  python vesta_colorize_sitet_tricolor.py input.vesta -n 30 -o out.vesta
"""

import argparse
import os
from pathlib import Path
import re
from typing import Tuple, List

HEADER_RE = re.compile(r'^[A-Z][A-Z0-9 _-]*\s*$')
TRIPLET_RE = re.compile(r'(-?\d+)\s+(-?\d+)\s+(-?\d+)')
# 形如：  <编号> <标签> <半径> <数字...>
LINE_RE = re.compile(r'(^\s*\d+\s+\S+\s+)([0-9]*\.?[0-9]+)(\s+.*)$')


def lerp(a: float, b: float, t: float) -> int:
    """线性插值 a -> b，在 [0,1] 上，返回整数（用于 RGB 分量）"""
    return int(round(a + (b - a) * t))


def make_linear_gradient(start: Tuple[int, int, int],
                         end: Tuple[int, int, int],
                         n: int) -> List[Tuple[int, int, int]]:
    """简单的两端线性渐变，共 n 个点"""
    if n <= 1:
        return [end]
    out = []
    for i in range(n):
        t = i / (n - 1)
        out.append((
            lerp(start[0], end[0], t),
            lerp(start[1], end[1], t),
            lerp(start[2], end[2], t),
        ))
    return out


def make_red_white_blue_gradient(n: int) -> List[Tuple[int, int, int]]:
    """
    生成长度为 n 的红->白->蓝渐变：
      - 前半段：红(255,0,0) -> 白(255,255,255)
      - 后半段：白(255,255,255) -> 蓝(0,0,255)
    """
    if n <= 0:
        return []

    red = (0, 100, 200) #实际上是blue
    white = (255, 255, 255)
    blue = (255, 0, 0) #实际上是red

    if n == 1:
        # 只有一个点，直接给白色
        return [white]
    if n == 2:
        # 两个点，直接红、蓝，中间白省略
        return [red, blue]

    # n >= 3 的情况：前半和后半
    # 例如 n=3: n1=1, n2=2 -> [红] + [白,蓝] = 红,白,蓝
    n1 = n // 2           # 红->白 的点数
    n2 = n - n1           # 白->蓝 的点数

    grad1 = make_linear_gradient(red, white, n1) if n1 > 0 else []
    grad2 = make_linear_gradient(white, blue, n2)

    if n1 > 0:
        return grad1 + grad2
    else:
        return grad2


def find_sitet_span(lines: List[str]) -> Tuple[int, int]:
    """找到 SITET 段的起止行索引 [start, end)"""
    start = None
    for i, ln in enumerate(lines):
        if ln.strip() == "SITET":
            start = i + 1
            break
    if start is None:
        raise RuntimeError("未找到 SITET 段；请确认是含原子外观信息的 .vesta 文件。")
    end = len(lines)
    for j in range(start, len(lines)):
        t = lines[j].strip()
        if t and HEADER_RE.match(t):
            end = j
            break
    return start, end


def colorize_last_n(in_path: str, out_path: str, n_last: int
                    ) -> Tuple[int, int, int]:
    """
    将 SITET 段中“最后 n_last 行原子条目”改为红->白->蓝渐变。
    返回 (edited, used, total)：
      edited: 实际成功修改颜色的行数
      used:   实际参与渐变的条目数（min(n_last, total)）
      total:  SITET 中原子条目总数
    """
    lines = Path(in_path).read_text(encoding="utf-8", errors="ignore").splitlines(True)
    s0, s1 = find_sitet_span(lines)

    # SITET 数据行（排除空行/段落名）
    data_idx = [
        i for i in range(s0, s1)
        if lines[i].strip() and not HEADER_RE.match(lines[i].strip())
    ]
    total = len(data_idx)
    if total == 0:
        raise RuntimeError("SITET 段为空。")

    n_last = max(1, min(n_last, total))
    grad = make_red_white_blue_gradient(n_last)
    targets = data_idx[-n_last:]

    edited = 0
    for idx, rgb in zip(targets, grad):
        raw = lines[idx].rstrip("\n")
        m = LINE_RE.match(raw)
        if m:
            prefix, size_val, tail = m.groups()
            triplets = TRIPLET_RE.findall(tail)
            new_rgb = f"{rgb[0]} {rgb[1]} {rgb[2]}"
            if len(triplets) >= 2:
                # 仅替换前两组三元色（通常为球体颜色和键颜色）
                tail = TRIPLET_RE.sub(new_rgb, tail, count=1)
                tail = TRIPLET_RE.sub(new_rgb, tail, count=1)
                lines[idx] = prefix + size_val + tail + "\n"
                edited += 1
            elif len(triplets) == 1:
                tail = TRIPLET_RE.sub(new_rgb, tail, count=1)
                lines[idx] = prefix + size_val + tail + "\n"
                edited += 1
            else:
                # 未检测到三元色，跳过
                pass
        else:
            # 兜底：直接在整行里替换两次三元色
            new_rgb = f"{rgb[0]} {rgb[1]} {rgb[2]}"
            tmp = raw
            if TRIPLET_RE.search(tmp):
                tmp = TRIPLET_RE.sub(new_rgb, tmp, count=1)
                if TRIPLET_RE.search(tmp):
                    tmp = TRIPLET_RE.sub(new_rgb, tmp, count=1)
                lines[idx] = tmp + "\n"
                edited += 1

    Path(out_path).write_text("".join(lines), encoding="utf-8")
    return edited, n_last, total


def main():
    ap = argparse.ArgumentParser(description="在 .vesta 的 SITET 段将最后 N 个原子设为红->白->蓝渐变")
    ap.add_argument("input", help="输入 .vesta 文件")
    ap.add_argument("-n", "--num", type=int, required=True,
                    help="需要修改的原子数（从最后开始计）")
    ap.add_argument("-o", "--output",
                    help="输出 .vesta（默认 input_basename_colored.vesta）")
    args = ap.parse_args()

    in_path = args.input
    if not os.path.isfile(in_path):
        raise SystemExit(f"找不到输入文件：{in_path}")
    out_path = args.output or (os.path.splitext(in_path)[0] + "_colored.vesta")

    edited, used, total = colorize_last_n(in_path, out_path, args.num)
    print(f"完成：在 {total} 个 SITET 条目中，已选取最后 {used} 个做红->白->蓝渐变；成功改色 {edited} 行。")
    print(f"输出文件：{out_path}")


if __name__ == "__main__":
    main()

