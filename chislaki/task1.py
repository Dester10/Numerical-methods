import numpy as np
import matplotlib.pyplot as plt
import math

# Создаем массив значений x от -10 до 10
x = np.linspace(-10, 10, 1000)

# Вычисляем значения функции x^3 + 6x + 7
y = x**3 + 6*x + 7

# Строим график
plt.figure(figsize=(10, 6))
plt.plot(x, y, label='y = x^3 + 6x + 7')
plt.xlabel('x')
plt.ylabel('y')
plt.title('График функции y = x^3 + 6x + 7')
plt.axhline(0, color='black',linewidth=0.5)
plt.axvline(0, color='black',linewidth=0.5)
plt.grid(True)
plt.legend()
plt.show()

def f(x):
    return x**3 + 6*x + 7

def bisection_method(a, b, e):
    if f(a) * f(b) >= 0:
        print("Bisection method fails.")
        return None
    while (b - a) >= e:
        c = (a + b) / 2
        if f(c) == 0:
            return c
        elif f(c) * f(a) < 0:
            b = c
        else:
            a = c
    
    return (a + b) / 2

# Найдем корни уравнения на разных интервалах
intervals = [(-3, -2), (0, 1), (1, 2)]  # Примерные интервалы для поиска корней
e = 0.001

for a, b in intervals:
    root = bisection_method(a, b, e)
    if root is not None:
        print(f"Корень уравнения на интервале ({a}, {b}): {root}")
