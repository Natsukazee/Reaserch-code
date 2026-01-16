# -*- coding: utf-8 -*-
"""
Created on Mon Jul 22 16:28:39 2024

@author: 86132
"""
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy import integrate

# Define parameters
A  = 0.1
b1 = 0.75
b2 = 60
b3 = 25
b4 = 6.21
b5 = 1.5708
epsilon = 5.7
epsilon_0 = 8.854e-12  # 真空中的介电常数 (F/m)
c = 3.00e8  # 光速 (m/s)

# Define time range
t = np.linspace(0, 120, 2000)

# Calculate function values with new parameters
f = - np.sqrt(3) * A * 1.244 * b1 * np.exp(-(t - b2)**2 / b3**2) * np.sin(b4 * t + b5) # V/A
#f2 = - np.sqrt(3) * A * 1.244 * b1 *  np.exp(-(t - b2)**2 / b3**2) * np.sin(b4 * t + b5) * 5.16129 * 10**(-15) * 10**(12) * 1000
I = 0.5 * epsilon_0 *epsilon* c * f**2 * 1e+1#辐照强度 单位 j/(cm^2*fs)
# Perform the numerical integration over t from 0 to 50 fs
#integral_value = integrate.simpson(abs(f2),x=t)
energy_per_cm2 = integrate.simpson(I, x=t)  # 单位：j/cm^2
# Print the maximum value of f
print(f"Maximum value of f(t): {np.max(f)}")
print(f"Total energy per cm²: {energy_per_cm2:.2e} J/cm²")

# Print the integral value
#print(f"Integral of f2(t) from 0 to 100 fs: {integral_value}")

# Path to desktop
desktop_path = os.path.expanduser("~/Desktop")

# Save t and f values to a single file on the desktop
file_path = os.path.join(desktop_path, f'{b1}-E-TDDFT_TIME.txt')
data = np.column_stack((t, f))
np.savetxt(file_path, data, header='t f', comments='')

# Plot the function with new parameters
# Plot the function with new parameters
plt.figure(figsize=(10, 6))
plt.plot(t, f, label=r'$f(t) = 0.75 e^{-(t - 60)^2 / 25^2} \sin(6.21 t + 1.5708)$')
plt.title('Function Plot with New Parameters')
plt.xlabel('Time $t$')
plt.ylabel('$f(t)$')

# 设置横纵坐标范围（要放在 plt.show() 之前）
plt.xlim(0, 120)  # 设置横坐标范围
plt.ylim(-1.1, 1.1)  # 设置纵坐标范围

# 显示图例和网格
plt.legend()
plt.grid(True)

# 显示图像
plt.show()




