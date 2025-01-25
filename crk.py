import numpy as np

def generate_dkp(n, scale):
    """
    Generuje losowy dyskretny problem plecakowy (DKP).

    Argumenty:
        n (int): liczba przedmiotów
        scale (float): współczynnik skalujący trudność problemu

    Zwraca:
        items (np.ndarray): macierz (n x 2), gdzie każda para to [wartość, waga]
        C (int): pojemność plecaka
    """
    items = np.ceil(scale * np.random.rand(n, 2)).astype("int32")
    C = int(np.ceil(0.5 * 0.5 * n * scale))
    return items, C

def solve_dkp(items, C):
    """
    Rozwiązuje problem DKP metodą dynamicznego programowania.

    Argumenty:
        items (np.ndarray): macierz (n x 2), gdzie każda para to [wartość, waga]
        C (int): pojemność plecaka

    Zwraca:
        max_value (int): maksymalna wartość możliwa do osiągnięcia
        solution (list): lista bitowa reprezentująca wybrane przedmioty
    """
    n = items.shape[0]
    values = items[:, 0]
    weights = items[:, 1]

    # Tworzymy tablicę dynamicznego programowania
    dp = np.zeros((n + 1, C + 1), dtype="int32")

    # Wypełnianie tablicy DP
    for i in range(1, n + 1):
        for w in range(C + 1):
            if weights[i - 1] <= w:
                dp[i, w] = max(dp[i - 1, w], dp[i - 1, w - weights[i - 1]] + values[i - 1])
            else:
                dp[i, w] = dp[i - 1, w]

    # Znajdowanie rozwiązania (śledzenie ścieżki)
    max_value = dp[n, C]
    solution = [0] * n
    w = C
    for i in range(n, 0, -1):
        if dp[i, w] != dp[i - 1, w]:
            solution[i - 1] = 1
            w -= weights[i - 1]

    return max_value, solution

# Przykład użycia
if __name__ == "__main__":
    n = 5
    scale = 10
    items, C = generate_dkp(n, scale)
    print("Przedmioty (wartość, waga):")
    print(items)
    print("Pojemność plecaka:", C)

    max_value, solution = solve_dkp(items, C)
    print("Maksymalna wartość:", max_value)
    print("Rozwiązanie (bitowe):", solution)
