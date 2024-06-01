import math
import numpy as np
import matplotlib.pyplot as plt

# Определяем функцию уравнения
def f(x):
    return x**3 - 3*x**2 - 4*x + 2

# Производная функции для метода Ньютона
def df(x):
    return 3*x**2 - 6*x - 4

# Создаем массив значений x
x = np.linspace(-3, 3, 500)  # от -3 до 3 с шагом 0.01

# Вычисляем соответствующие значения y
y = f(x)

# Строим график
plt.figure(figsize=(10, 6))
plt.plot(x, y, label='$x^3 - 3x^2 - 4x + 2$')
plt.axhline(y=0, color='k', linestyle='--', linewidth=0.8)  # горизонтальная линия y=0
plt.axvline(x=0, color='k', linestyle='--', linewidth=0.8)  # вертикальная линия y=0
plt.xlabel('x')
plt.ylabel('f(x)')
plt.title('График уравнения $x^3 - 3x^2 - 4x + 2$')
plt.grid(True)
plt.legend()
plt.show()

# Метод простой итерации (преобразование функции)
def g(x):
    return (x**3 - 4*x + 2) / 3

def simple_iteration_method(x0, e):
    x_prev = x0
    x_next = g(x_prev)
    
    while abs(x_next - x_prev) >= e:
        x_prev = x_next
        x_next = g(x_prev)
    
    return x_next

# Метод Ньютона
def newton_method(x0, e):
    x_prev = x0
    x_next = x_prev - f(x_prev) / df(x_prev)
    
    while abs(x_next - x_prev) >= e:
        x_prev = x_next
        x_next = x_prev - f(x_prev) / df(x_prev)
    
    return x_next

# Решение методом простой итерации
x0_simple = 0.5  # Начальное приближение (может быть изменено в зависимости от графика)
x_simple = simple_iteration_method(x0_simple, 0.001)
print(f"Корень уравнения методом простой итерации: {x_simple}")

# Решение методом Ньютона
x0_newton = 0.5  # Начальное приближение (может быть изменено в зависимости от графика)
x_newton = newton_method(x0_newton, 0.001)
print(f"Корень уравнения методом Ньютона: {x_newton}")