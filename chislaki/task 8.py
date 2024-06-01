import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad

# --------------- Решение дифференциального уравнения ---------------

# Определяем дифференциальное уравнение
def f(x, y):
    return (x - 1) * y / x**2

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
x0 = 1
y0 = np.e
x_end = 2
h = 0.05

# Решаем дифференциальное уравнение
x_euler, y_euler = euler(f, x0, y0, h, x_end)
x_rk4, y_rk4 = rk4(f, x0, y0, h, x_end)

# Построим графики
plt.figure(figsize=(8, 6))
plt.plot(x_euler, y_euler, label='Euler')
plt.plot(x_rk4, y_rk4, label='Runge-Kutta 4')
plt.legend()
plt.xlabel('x')
plt.ylabel('y')
plt.title('Решения дифференциального уравнения')
plt.grid()
plt.show()

# --------------- Решение интеграла ---------------

# Определение функции
def integrand(x):
    return np.tan(x**2) / (2 * x**2 - 1)

# Пределы интегрирования
a = 0.4
b = 0.8

# Метод прямоугольников
def rectangle_method(f, a, b, n):
    h = (b - a) / n
    result = 0
    for i in range(n):
        result += f(a + i * h + h / 2)
    return result * h

# Метод трапеций
def trapezoidal_method(f, a, b, n):
    h = (b - a) / n
    result = 0.5 * (f(a) + f(b))
    for i in range(1, n):
        result += f(a + i * h)
    return result * h

# Метод Симпсона
def simpson_method(f, a, b, n):
    h = (b - a) / n
    result = f(a) + f(b)
    for i in range(1, n, 2):
        result += 4 * f(a + i * h)
    for i in range(2, n-1, 2):
        result += 2 * f(a + i * h)
    return result * h / 3

# Вычисления
n = 10000  # Количество шагов
rectangle_result = rectangle_method(integrand, a, b, n)
trapezoidal_result = trapezoidal_method(integrand, a, b, n)
simpson_result = simpson_method(integrand, a, b, n)
exact_result, _ = quad(integrand, a, b)

print(f"Метод прямоугольников: {rectangle_result}")
print(f"Метод трапеций: {trapezoidal_result}")
print(f"Метод Симпсона: {simpson_result}")
print(f"Точное значение: {exact_result}")
print(f"Отклонение метода прямоугольников: {abs(rectangle_result - exact_result)}")
print(f"Отклонение метода трапеций: {abs(trapezoidal_result - exact_result)}")
print(f"Отклонение метода Симпсона: {abs(simpson_result - exact_result)}")