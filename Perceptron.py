import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 1. Создание исходного набора данных
num_points = 1000
x1 = np.random.uniform(0, 2 * np.pi, num_points)
x2 = np.random.uniform(-1, 1, num_points)
X = np.column_stack((np.ones(num_points), x1, x2))

y = np.where(np.abs(np.sin(x1)) > np.abs(x2), -1, 1)

# Визуализация исходного набора данных
plt.figure(figsize=(8, 6))
plt.scatter(x1[y == -1], x2[y == -1], edgecolor='green', facecolor='none', marker='o', s=10, label='Class -1')
plt.scatter(x1[y == 1], x2[y == 1], edgecolor='blue', facecolor='none', marker='o', s=10, label='Class 1')
plt.xlabel('$x_1$')
plt.ylabel('$x_2$')
plt.title('Исходный набор данных')
plt.legend()
plt.grid()
plt.show()

# 2. Нормализация данных
x1_normalized = (x1 - x1.min()) / (x1.max() - x1.min()) * 2 - 1
x2_normalized = x2  # x2 уже в диапазоне [-1, 1]
X_normalized = np.column_stack((np.ones(num_points), x1_normalized, x2_normalized))

# 3. Поднятие размерности пространства (Пространство признаков)
num_centers = 100
centers = np.random.uniform(-1, 1, (num_centers, 2))  # Случайное расположение центров
sigma = 0.3  # Параметр ядра Гаусса

# Преобразование данных в пространство признаков
Z = np.zeros((num_points, num_centers))
for i in range(num_points):
    for j in range(num_centers):
        distance_squared = (x1_normalized[i] - centers[j, 0])**2 + (x2_normalized[i] - centers[j, 1])**2
        Z[i, j] = np.exp(-distance_squared / (2 * sigma**2))

# Добавление bias в матрицу Z
Z = np.column_stack((np.ones(num_points), Z))

# 4. Реализация алгоритма перцептрона
weights = np.zeros(Z.shape[1])
max_iterations = 1000
learning_rate = 0.01

for iteration in range(max_iterations):
    errors = 0
    for i in range(num_points):
        prediction = np.sign(np.dot(weights, Z[i]))
        if prediction != y[i]:
            weights += learning_rate * y[i] * Z[i]
            errors += 1
    if errors == 0:
        break

# 5. Визуализация результатов
x1_mesh, x2_mesh = np.meshgrid(np.linspace(-1, 1, 200), np.linspace(-1, 1, 200))
x1_flat, x2_flat = x1_mesh.ravel(), x2_mesh.ravel()

Z_mesh = np.zeros((len(x1_flat), num_centers))
for i in range(len(x1_flat)):
    for j in range(num_centers):
        distance_squared = (x1_flat[i] - centers[j, 0])**2 + (x2_flat[i] - centers[j, 1])**2
        Z_mesh[i, j] = np.exp(-distance_squared / (2 * sigma**2))

Z_mesh = np.column_stack((np.ones(len(x1_flat)), Z_mesh))
predictions = np.sign(Z_mesh.dot(weights)).reshape(x1_mesh.shape)

plt.figure(figsize=(8, 6))
plt.contourf(x1_mesh, x2_mesh, predictions, levels=[-1, 0, 1], colors=['red', 'yellow'], alpha=0.5)
plt.scatter(x1_normalized[y == -1], x2_normalized[y == -1], edgecolor='green', facecolor='none', marker='o', s=10, label='Class -1')
plt.scatter(x1_normalized[y == 1], x2_normalized[y == 1], edgecolor='blue', facecolor='none', marker='o', s=10, label='Class 1')
plt.scatter(centers[:, 0], centers[:, 1], c='black', s=50, label='Centers')  # Добавление центров
plt.xlabel('$x_1$')
plt.ylabel('$x_2$')
plt.title('Результаты классификации')
plt.legend()
plt.grid()
plt.show()

# 6. Визуализация весовой суммы в 3D
weighted_sum = Z_mesh.dot(weights).reshape(x1_mesh.shape)
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(x1_mesh, x2_mesh, weighted_sum, cmap='coolwarm', edgecolor='none', alpha=0.8)
ax.set_xlabel('$x_1$')
ax.set_ylabel('$x_2$')
ax.set_zlabel('Weighted sum')
ax.set_title('Весовая сумма (3D визуализация)')
plt.show()

# 7. Визуализация контурного графика весовой суммы
plt.figure(figsize=(8, 6))
contour_levels = np.linspace(weighted_sum.min(), weighted_sum.max(), 50)  # Увеличение числа варствиц
plt.contour(x1_mesh, x2_mesh, weighted_sum, levels=contour_levels, cmap='coolwarm')
plt.colorbar(label='Weighted sum')
plt.xlabel('$x_1$')
plt.ylabel('$x_2$')
plt.title('Контурный график весовой суммы')
plt.grid()
plt.show()
