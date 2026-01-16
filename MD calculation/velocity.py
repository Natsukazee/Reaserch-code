import math
import re

output_file = 'output-velocity.txt'
Z = 6  # 原子序号
A = 2 * Z  # 质量数（近似）

# 常量前计算好
factor = 14.51 * A

with open(output_file, 'w') as o:
    with open('MOVEMENT.txt', 'r') as f:
        i = 1  # 帧编号
        j = 1  # 行编号
        line = f.readline()

        while line:
            if j == 294 * (i - 1) + 19 + 198:  # 第 i 帧中第 217 行（原子编号19）
                print(i, end=' ', file=o)  # 输出时间步编号
                newline = line.replace('\n', ' ')
                print(newline, end=' ', file=o)

                # 提取浮点数（速度）
                numbers = re.findall(r'\b\d+\.\d+\b', line)
                if len(numbers) >= 3:
                    vx, vy, vz = map(float, numbers[:3])
                    v2 = vx ** 2 + vy ** 2 + vz ** 2
                    Ek = factor * v2
                    print(f"{Ek:.6f}", file=o)  # 保留小数位输出
                else:
                    print("NaN", file=o)  # 速度解析失败时
                i += 1

            j += 1
            line = f.readline()

