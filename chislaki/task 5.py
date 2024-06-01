import numpy as np
import matplotlib.pyplot as plt

# Задаем коэффициенты системы уравнений
A = np.array([[-11, -8, 0, 0, 0],
              [9, -17, 1, 0, 0],
              [0, -4, 20, 9, 0],
              [0, 0, -4, -14, 3],
              [0, 0, 0, -6, 14]])

D = np.diag(A)
L = np.diag(A, k=-1)
U = np.diag(A, k=1)

# Решение системы уравнений методом прогонки
def tridiagonal_solver(D, L, U, b):
    n = len(b)
    alpha = np.zeros(n-1)
    beta = np.zeros(n)
    x = np.zeros(n)

    alpha[0] = U[0] / D[0]
    beta[0] = b[0] / D[0]
    
    # Прямой ход для C(i)
    for i in range(1, n-1):
        alpha[i] = U[i] / (D[i] - L[i-1]*alpha[i-1])
    
    # Прямой ход для D(i)
    for i in range(1, n):
        beta[i] = (b[i] - L[i-1]*beta[i-1]) / (D[i] - L[i-1]*alpha[i-1])
    
    # Обратный ход
    x[n-1] = beta[n-1]
    for i in range(n-2, -1, -1):
        x[i] = beta[i] - alpha[i]*x[i+1]
    
    return x

# Вектор правой части
b = np.array([99, -75, 66, 54, 8])

# Решение системы уравнений
solution = tridiagonal_solver(D, L, U, b)

print("Решение системы уравнений:")
for i in range(len(solution)):
    print(f"x{i+1} =", solution[i])

# Построение графика
plt.figure(figsize=(10, 5))
plt.plot(range(1, len(solution) + 1), solution, marker='o', linestyle='-', color='b')
plt.title('Решение системы уравнений методом прогонки')
plt.xlabel('Индекс переменной')
plt.ylabel('Значение переменной')
plt.grid(True)
plt.show()