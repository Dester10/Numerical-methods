import numpy as np
import matplotlib.pyplot as plt

# Определяем дифференциальное уравнение
def f(x, y):
    return y**2 / x**2

# Метод Эйлера-Коши
def euler(f, x0, y0, h, x_end):
    x = np.arange(x0, x_end + h, h)
    y = np.zeros_like(x)
    y[0] = y0
    for i in range(1, len(x)):
        y[i] = y[i - 1] + h * f(x[i - 1], y[i - 1])
    return x, y

# Метод Рунге-Кутта 4-го порядка
def rk4(f, x0, y0, h, x_end):
    x = np.arange(x0, x_end + h, h)
    y = np.zeros_like(x)
    y[0] = y0
    for i in range(1, len(x)):
        k1 = f(x[i - 1], y[i - 1])
        k2 = f(x[i - 1] + h / 2, y[i - 1] + k1 * h / 2)
        k3 = f(x[i - 1] + h / 2, y[i - 1] + k2 * h / 2)
        k4 = f(x[i - 1] + h, y[i - 1] + k3 * h)
        y[i] = y[i - 1] + h * (k1 + 2 * k2 + 2 * k3 + k4) / 6
    return x, y

# Начальные условия
x0 = 0.1  # Начальная точка чуть больше нуля, чтобы избежать деления на ноль
y0 = 1
x_end = 2
h = 0.1

# Решаем дифференциальное уравнение
x_euler, y_euler = euler(f, x0, y0, h, x_end)
x_rk4, y_rk4 = rk4(f, x0, y0, h, x_end)

# Построим графики
plt.figure(figsize=(10, 6))
plt.plot(x_euler, y_euler, label='Метод Эйлера-Коши')
plt.plot(x_rk4, y_rk4, label='Метод Рунге-Кутта 4-го порядка')

# Точное решение уравнения
x_exact = np.linspace(x0, x_end, 300)
y_exact = x_exact**2 / (1 - np.log(x_exact))
plt.plot(x_exact, y_exact, label='Точное решение', linestyle='dashed')

plt.legend()
plt.xlabel('x')
plt.ylabel('y')
plt.title('Решения дифференциального уравнения')
plt.grid()
plt.show()