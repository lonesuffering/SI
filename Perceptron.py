import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 1. Tworzenie początkowego zbioru danych
num_points = 1000
x1 = np.random.uniform(0, 2 * np.pi, num_points)
x2 = np.random.uniform(-1, 1, num_points)
X = np.column_stack((np.ones(num_points), x1, x2))

y = np.where(np.abs(np.sin(x1)) > np.abs(x2), -1, 1)

# Wizualizacja początkowego zbioru danych
plt.figure(figsize=(8, 6))
plt.scatter(x1[y == -1], x2[y == -1], edgecolor='green', facecolor='none', marker='o', s=10, label='Klasa -1')
plt.scatter(x1[y == 1], x2[y == 1], edgecolor='blue', facecolor='none', marker='o', s=10, label='Klasa 1')
plt.xlabel('$x_1$')
plt.ylabel('$x_2$')
plt.title('Początkowy zbiór danych')
plt.legend()
plt.grid()
plt.show()

# 2. Normalizacja danych
x1_normalized = (x1 - x1.min()) / (x1.max() - x1.min()) * 2 - 1
x2_normalized = x2  # x2 jest już w zakresie [-1, 1]
X_normalized = np.column_stack((np.ones(num_points), x1_normalized, x2_normalized))

# 3. Podniesienie wymiarowości przestrzeni (Przestrzeń cech)
num_centers = 100
centers = np.random.uniform(-1, 1, (num_centers, 2))  # Losowe rozmieszczenie centrów
sigma = 0.3  # Parametr jądra Gaussa

# Przekształcenie danych do przestrzeni cech
Z = np.zeros((num_points, num_centers))
for i in range(num_points):
    for j in range(num_centers):
        distance_squared = (x1_normalized[i] - centers[j, 0])**2 + (x2_normalized[i] - centers[j, 1])**2
        Z[i, j] = np.exp(-distance_squared / (2 * sigma**2))

# Dodanie bias do macierzy Z
Z = np.column_stack((np.ones(num_points), Z))

# 4. Implementacja algorytmu perceptronu
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

# 5. Wizualizacja wyników
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
plt.scatter(x1_normalized[y == -1], x2_normalized[y == -1], edgecolor='green', facecolor='none', marker='o', s=10, label='Klasa -1')
plt.scatter(x1_normalized[y == 1], x2_normalized[y == 1], edgecolor='blue', facecolor='none', marker='o', s=10, label='Klasa 1')
plt.scatter(centers[:, 0], centers[:, 1], c='black', s=50, label='Centra')  # Dodanie centrów
plt.xlabel('$x_1$')
plt.ylabel('$x_2$')
plt.title('Wyniki klasyfikacji')
plt.legend()
plt.grid()
plt.show()

# 6. Wizualizacja sumy ważonej w 3D
weighted_sum = Z_mesh.dot(weights).reshape(x1_mesh.shape)
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(x1_mesh, x2_mesh, weighted_sum, cmap='coolwarm', edgecolor='none', alpha=0.8)
ax.set_xlabel('$x_1$')
ax.set_ylabel('$x_2$')
ax.set_zlabel('Suma ważona')
ax.set_title('Suma ważona (wizualizacja 3D)')
plt.show()

# 7. Wizualizacja wykresu warstwicowego sumy ważonej
plt.figure(figsize=(8, 6))
contour_levels = np.linspace(weighted_sum.min(), weighted_sum.max(), 50)  # Większa liczba warstwic
plt.contour(x1_mesh, x2_mesh, weighted_sum, levels=contour_levels, cmap='coolwarm')
plt.colorbar(label='Suma ważona')
plt.xlabel('$x_1$')
plt.ylabel('$x_2$')
plt.title('Wykres warstwicowy sumy ważonej')
plt.grid()
plt.show()

Wpływ współczynnika uczenia na liczbę iteracji:
Współczynnik uczenia: 0.001, Liczba iteracji: 1000
Współczynnik uczenia: 0.010, Liczba iteracji: 480
Współczynnik uczenia: 0.100, Liczba iteracji: 60
Współczynnik uczenia: 0.500, Liczba iteracji: 20

Wpływ marginesu między klasami na liczbę iteracji:
Margines: 0.10, Liczba iteracji: 200
Margines: 0.30, Liczba iteracji: 150
Margines: 0.50, Liczba iteracji: 80
Margines: 0.70, Liczba iteracji: 40
Margines: 0.90, Liczba iteracji: 20

