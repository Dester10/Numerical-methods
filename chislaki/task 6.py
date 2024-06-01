import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Заданные данные
x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
y = np.array([2.3, 3.71, 4.8, 5.9, 6.3, 6.25, 5.87, 4.82, 3.7, 2.2])

# Линейная регрессия: y = kx + b
def linear_func(x, k, b):
    return k * x + b

# вычисляем параметры
params_lin, _ = curve_fit(linear_func, x, y)
k_lin, b_lin = params_lin

# Квадратичная регрессия: y = a2x^2 + a1x + a0
def quadratic_func(x, a2, a1, a0):
    return a2 * x**2 + a1 * x + a0

# Параметры для квадратичной регрессии 
params_quad, _ = curve_fit(quadratic_func, x, y)
a2_quad, a1_quad, a0_quad = params_quad

print("Параметры линейной регрессии:")
print(f"k = {k_lin}, b = {b_lin}")
print("\nПараметры квадратичной регрессии:")
print(f"a2 = {a2_quad}, a1 = {a1_quad}, a0 = {a0_quad}")

# Построение графиков
plt.figure(figsize=(12, 6))

plt.scatter(x, y, color='blue', label='Табличные данные')
plt.plot(x, linear_func(x, *params_lin), color='red', label=f'Линейная регрессия: y = {k_lin:.2f}x + {b_lin:.2f}')
plt.plot(x, quadratic_func(x, *params_quad), color='green', label=f'Квадратичная регрессия: y = {a2_quad:.2f}x^2 + {a1_quad:.2f}x + {a0_quad:.2f}')

plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid(True)
plt.show()