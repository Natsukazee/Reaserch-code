from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Tuple, List
import pandas as pd


@dataclass(frozen=True)
class Range1D:
    """一维闭区间 [low, high]；若 low > high，则按周期穿越处理：x>=low 或 x<=high。"""
    low: float
    high: float

    def mask(self, s: pd.Series) -> pd.Series:
        if self.low <= self.high:
            return (s >= self.low) & (s <= self.high)
        # 周期穿越选择（例如 0.9 到 0.1）
        return (s >= self.low) | (s <= self.high)


def read_out_vatom(filepath: str) -> pd.DataFrame:
    """
    读取 OUT.VATOM:
    期望数据行格式: Z x y z V
    会跳过无法解析的行（如表头）。
    """
    records: List[tuple] = []
    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        for line_no, line in enumerate(f, start=1):
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            try:
                z = int(parts[0])
                x = float(parts[1])
                y = float(parts[2])
                zc = float(parts[3])
                v = float(parts[4])
                records.append((z, x, y, zc, v))
            except ValueError:
                # 跳过表头/说明行等
                continue

    if not records:
        raise ValueError(f"未能从文件中解析到任何数据行：{filepath}")

    df = pd.DataFrame(records, columns=["Z", "x", "y", "z", "V"])
    return df


def average_potential(
    df: pd.DataFrame,
    atomic_numbers: Iterable[int],
    x_range: Optional[Range1D] = None,
    y_range: Optional[Range1D] = None,
    z_range: Optional[Range1D] = None,
) -> dict:
    """
    按原子序数与坐标范围筛选并计算势能统计量。
    返回包含 count/mean/std/min/max 等信息的字典。
    """
    atomic_numbers = list(atomic_numbers)
    if not atomic_numbers:
        raise ValueError("atomic_numbers 不能为空。")

    mask = df["Z"].isin(atomic_numbers)

    if x_range is not None:
        mask &= x_range.mask(df["x"])
    if y_range is not None:
        mask &= y_range.mask(df["y"])
    if z_range is not None:
        mask &= z_range.mask(df["z"])

    sub = df.loc[mask, ["Z", "x", "y", "z", "V"]].copy()

    if sub.empty:
        return {
            "count": 0,
            "mean": None,
            "std": None,
            "min": None,
            "max": None,
            "note": "筛选结果为空，请检查 Z 与坐标范围是否匹配。",
        }

    return {
        "count": int(sub.shape[0]),
        "mean": float(sub["V"].mean()),
        "std": float(sub["V"].std(ddof=1)) if sub.shape[0] > 1 else 0.0,
        "min": float(sub["V"].min()),
        "max": float(sub["V"].max()),
        "subset": sub,  # 如不想返回明细，可删掉这一项
    }


def _prompt_range(name: str) -> Optional[Range1D]:
    """
    交互式输入范围：留空则不限制该轴。
    输入示例：
      0 0.5
      0.9 0.1  (周期穿越)
    """
    raw = input(f"请输入 {name} 范围 (low high)，留空表示不限制：").strip()
    if not raw:
        return None
    parts = raw.split()
    if len(parts) != 2:
        raise ValueError(f"{name} 范围输入格式错误，应为两个数，例如：0 0.5")
    low, high = float(parts[0]), float(parts[1])
    return Range1D(low, high)


def _prompt_atomic_numbers() -> List[int]:
    """
    交互式输入原子序数：支持单个或多个。
    示例：
      32
      8 14 32
    """
    raw = input("请输入原子序数 Z（一个或多个，用空格分隔）：").strip()
    if not raw:
        raise ValueError("原子序数 Z 不能为空。")
    return [int(x) for x in raw.split()]


def main():
    filepath = "OUT.VATOM"  # 如果文件不在当前目录，改成完整路径

    df = read_out_vatom(filepath)
    print(f"已读取 {filepath}：共 {len(df)} 条原子记录。")
    print("列：Z, x, y, z, V (eV)")

    zs = _prompt_atomic_numbers()
    xr = _prompt_range("x")
    yr = _prompt_range("y")
    zr = _prompt_range("z")

    res = average_potential(df, atomic_numbers=zs, x_range=xr, y_range=yr, z_range=zr)

    print("\n=== 结果 ===")
    print(f"筛选到的原子数: {res['count']}")
    if res["count"] == 0:
        print(res.get("note", "无结果。"))
        return

    print(f"势能均值 (eV): {res['mean']:.10f}")
    print(f"势能标准差 (eV): {res['std']:.10f}")
    print(f"势能最小/最大 (eV): {res['min']:.10f} / {res['max']:.10f}")

    # 如果你想查看明细（例如前 10 行），取消注释：
    # print("\n筛选明细前10行：")
    # print(res["subset"].head(10).to_string(index=False))


if __name__ == "__main__":
    main()
