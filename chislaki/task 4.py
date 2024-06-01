import numpy as np
import matplotlib.pyplot as plt

# Определяем функции системы уравнений
def f(x, y):
    return x**2 + y**2 - 9

def g(x, y):
    return y - np.exp(x)

def df_dx(x, y):
    return 2*x

def df_dy(x, y):
    return 2*y

def dg_dx(x, y):
    return -np.exp(x)

def dg_dy(x, y):
    return 1

# Метод Ньютона для системы уравнений
def newton_method(f, g, df_dx, df_dy, dg_dx, dg_dy, x0, y0, tol=1e-4, max_iter=1000):
    history = []
    for i in range(max_iter):
        F = np.array([f(x0, y0), g(x0, y0)])
        J = np.array([[df_dx(x0, y0), df_dy(x0, y0)], [dg_dx(x0, y0), dg_dy(x0, y0)]])
        delta = np.linalg.solve(J, -F)
        x0 += delta[0]
        y0 += delta[1]
        history.append((x0, y0))
        if np.linalg.norm(delta) < tol:
            print(f"Метод Ньютона сошелся за {i+1} итераций.")
            return x0, y0, history
    return x0, y0, history

# Метод простых итераций для системы уравнений
def simple_iteration_method(x0, y0, tol=1e-4, max_iter=5000):
    history = []
    for i in range(max_iter):
        x_new = x0  # фиксация значения x
        y_new = np.exp(x_new)
        x_new = np.sqrt(9 - y_new**2)
        history.append((x_new, y_new))
        if np.abs(x_new - x0) < tol and np.abs(y_new - y0) < tol:
            print(f"Метод простых итераций сошелся за {i+1} итераций.")
            return x_new, y_new, history
        x0 = x_new
        y0 = y_new
    return x_new, y_new, history

# Решение системы уравнений методом Ньютона
x_newton, y_newton, history_newton = newton_method(f, g, df_dx, df_dy, dg_dx, dg_dy, 1.5, 2.0)

# Решение системы уравнений методом простых итераций
x_simple_iteration, y_simple_iteration, history_simple_iteration = simple_iteration_method(1.5, 2.0)

print("Метод Ньютона:")
print("x =", x_newton)
print("y =", y_newton)

print("\nМетод простых итераций:")
print("x =", x_simple_iteration)
print("y =", y_simple_iteration)

# Построение графиков
x = np.linspace(-3, 3, 400)
y1 = np.sqrt(9 - x**2)
y2 = np.exp(x)

plt.figure(figsize=(12, 6))
plt.plot(x, y1, label='x^2 + y^2 = 9')
plt.plot(x, y2, label='y = exp(x)')
plt.scatter([x_newton], [y_newton], color='red', s=100, label='Newton solution')
plt.scatter([x_simple_iteration], [y_simple_iteration], color='green', s=100, label='Simple iteration solution')
plt.xlabel('x')
plt.ylabel('y')
plt.axhline(0, color='black',linewidth=0.5)
plt.axvline(0, color='black',linewidth=0.5)
plt.grid(color = 'gray', linestyle = '--', linewidth = 0.5)
plt.legend()
plt.title("Solutions of the system of equations using Newton and Simple Iteration Methods")
plt.show()

def plot_iterations(history, method_name):
    history = np.array(history)
    iterations = np.arange(len(history))
    
    plt.figure(figsize=(10, 6))
    plt.plot(iterations, history[:, 0], label='x')
    plt.plot(iterations, history[:, 1], label='y')
    
    plt.xlabel('Iteration')
    plt.ylabel('Value')
    plt.title(f'Convergence of {method_name}')
    plt.legend()
    plt.grid(True)
    plt.show()

# Построение графиков итераций
plot_iterations(history_newton, "Newton Method")
plot_iterations(history_simple_iteration, "Simple Iteration Method")