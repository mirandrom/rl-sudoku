from environment import ConstraintSudoku
import numpy as np


def get_action(grid: np.ndarray):
    # heuristic policy based on Norvig blog post
    num_valid = [(i, j, n) for i, r in enumerate(grid.sum(axis=-1)) for j, n in enumerate(r) if n > 1]
    row, col, _ = min(num_valid, key=lambda x: x[-1])
    val = grid[row][col].nonzero()[0][0]
    return {'row': row, 'col': col, 'val': val}


def verify_win(grid: np.ndarray):
    # check correctness of completed game
    assert (grid.sum(axis=-1) == 1).all(), "Each cell must have one remaining valid value"
    for row in range(len(grid)):
        row_vals = set(grid[row].nonzero()[-1])
        assert len(row_vals) == len(grid), f"{row}th row has valid values {row_vals}"
    for col in range(len(grid)):
        col_vals = set(grid[:, col, :].nonzero()[-1])
        assert len(col_vals) == len(grid), f"{col}th col has valid values {col_vals}"
    d = int(len(grid) ** 0.5)
    for i in range(d):
        for j in range(d):
            i1 = i * d
            i2 = i1 + d
            j1 = j * d
            j2 = j1 + d
            sq_vals = set(grid[i1:i2, j1:j2].nonzero()[-1])
            assert len(sq_vals) == len(grid), f"{i},{j}th square has valid values {sq_vals}"


csdku = ConstraintSudoku()
num_trials = 10
for i in range(num_trials):
    grid = csdku.reset()
    total_reward = 0
    while True:
        action = get_action(grid)
        grid, reward, finished, meta = csdku.step(action)
        print(reward)
        total_reward += reward
        if finished:
            verify_win(csdku.grid)
            print(f"Trial {i} reward: {total_reward}")
            print(csdku.render())
            break