import numpy as np
import matplotlib.pyplot as plt

# Заданная функция для интегрирования
def f(x):
    return 1 / np.sqrt(2 * x**2 + 1.3)

# Функция для вычисления интеграла методом прямоугольников
def rectangle_rule(f, a, b, n):
    h = (b - a) / n
    integral = 0
    for i in range(n):
        integral += f(a + i * h)
    integral *= h
    return integral

# Функция для вычисления интеграла методом трапеций
def trapezoidal_rule(f, a, b, n):
    h = (b - a) / n
    integral = 0.5 * (f(a) + f(b))
    for i in range(1, n):
        integral += f(a + i * h)
    integral *= h
    return integral

# Функция для вычисления интеграла методом Симпсона
def simpsons_rule(f, a, b, n):
    h = (b - a) / n
    integral = f(a) + f(b)
    x = a
    for i in range(1, n):
        x += h
        if i % 2 == 0:
            integral += 2 * f(x)
        else:
            integral += 4 * f(x)
    integral *= h / 3
    return integral

# Точное значение интеграла (если возможно)
exact_integral = 0.4231953

# Вычисление интегралов с заданной точностью
tolerance = 0.0001

# Метод прямоугольников
n = 1
rectangle_integral = rectangle_rule(f, 1, 2, n)
rectangle_error = abs(exact_integral - rectangle_integral)
while rectangle_error > tolerance:
    n += 1
    rectangle_integral = rectangle_rule(f, 1, 2, n)
    rectangle_error = abs(exact_integral - rectangle_integral)

# Метод трапеций
n = 1
trapezoidal_integral = trapezoidal_rule(f, 1, 2, n)
trapezoidal_error = abs(exact_integral - trapezoidal_integral)
while trapezoidal_error > tolerance:
    n += 1
    trapezoidal_integral = trapezoidal_rule(f, 1, 2, n)
    trapezoidal_error = abs(exact_integral - trapezoidal_integral)

# Метод Симпсона
n = 1
simpsons_integral = simpsons_rule(f, 1, 2, n)
simpsons_error = abs(exact_integral - simpsons_integral)
while simpsons_error > tolerance:
    n += 1
    simpsons_integral = simpsons_rule(f, 1, 2, n)
    simpsons_error = abs(exact_integral - simpsons_integral)

# Вывод результатов
print(f"Точное значение интеграла: {exact_integral}")
print(f"Приближенное значение интеграла (метод прямоугольников): {rectangle_integral}, Погрешность: {rectangle_error}")
print(f"Приближенное значение интеграла (метод трапеций): {trapezoidal_integral}, Погрешность: {trapezoidal_error}")
print(f"Приближенное значение интеграла (метод Симпсона): {simpsons_integral}, Погрешность: {simpsons_error}")

# График функции для визуализации
x = np.linspace(1, 2, 100)
y = f(x)

plt.plot(x, y, label='f(x) = 1 / sqrt(2x^2 + 1.3)')
plt.fill_between(x, y, color='skyblue', alpha=0.5)
plt.xlabel('x')
plt.ylabel('f(x)')
plt.title('График функции')
plt.legend()
plt.show()