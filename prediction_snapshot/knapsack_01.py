"""
0-1 Knapsack Problem

Given n items with weights and values, pick a subset (each item at most once)
so that total weight <= capacity and total value is maximized.
"""


def knapsack_01(weights, values, capacity):
    """
    Return the maximum achievable value.

    Args:
        weights: list of item weights (positive ints)
        values: list of item values (non-negative ints)
        capacity: knapsack capacity (non-negative int)

    Time: O(n * capacity), Space: O(capacity)
    """
    if capacity < 0:
        raise ValueError("capacity must be non-negative")
    if len(weights) != len(values):
        raise ValueError("weights and values must have the same length")

    n = len(weights)
    dp = [0] * (capacity + 1)

    for i in range(n):
        w, v = weights[i], values[i]
        if w > capacity:
            continue
        for c in range(capacity, w - 1, -1):
            dp[c] = max(dp[c], dp[c - w] + v)

    return dp[capacity]


def knapsack_01_with_items(weights, values, capacity):
    """
    Return (max_value, selected_indices).

    Uses the same DP as knapsack_01, then backtracks to recover one optimal set.
    """
    if capacity < 0:
        raise ValueError("capacity must be non-negative")
    if len(weights) != len(values):
        raise ValueError("weights and values must have the same length")

    n = len(weights)
    dp = [[0] * (capacity + 1) for _ in range(n + 1)]

    for i in range(1, n + 1):
        w, v = weights[i - 1], values[i - 1]
        for c in range(capacity + 1):
            dp[i][c] = dp[i - 1][c]
            if w <= c:
                dp[i][c] = max(dp[i][c], dp[i - 1][c - w] + v)

    selected = []
    c = capacity
    for i in range(n, 0, -1):
        if dp[i][c] != dp[i - 1][c]:
            selected.append(i - 1)
            c -= weights[i - 1]

    selected.reverse()
    return dp[n][capacity], selected


if __name__ == "__main__":
    weights = [2, 3, 4, 5]
    values = [3, 4, 5, 6]
    capacity = 8

    max_value = knapsack_01(weights, values, capacity)
    max_value2, items = knapsack_01_with_items(weights, values, capacity)

    print(f"capacity = {capacity}")
    print(f"weights  = {weights}")
    print(f"values   = {values}")
    print(f"max value (space-optimized DP): {max_value}")
    print(f"max value (with backtrack):     {max_value2}")
    print(f"selected item indices:          {items}")
    print(
        f"selected weight / value: "
        f"{sum(weights[i] for i in items)} / {sum(values[i] for i in items)}"
    )
