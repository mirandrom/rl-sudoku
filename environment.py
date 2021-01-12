import gym
from gym import spaces
import numpy as np
import random
from collections import deque

from typing import Dict


class ConstraintSudoku(gym.Env):
    # Adapted from http://norvig.com/sudoku.html
    def __init__(self, d: int = 3, max_num_steps: int = 1e6):
        assert d % 2 == 1, "d must be odd"
        self.d = d
        self.D = d**2
        self.N = self.D * 2 - 1 # Number of solved cells at the start of a puzzle
        # A sudoku game can be represented as a DxD grid (D is typically 9)
        # and each cell in the grid can take one of D possible values.
        # At any given point, a cell may have one possible value (if its solved)
        # up to D possible values (if there are no constraints on the cell).
        # We express this as a binary DxDxD array where arr[i][j] is a
        # D-dimensional binary vector representing remaining possible values
        # for the cell in the i'th row and the j'th column
        self.observation_space = spaces.MultiBinary([self.D]*3)
        self.action_space = spaces.Dict({
            'row': spaces.Discrete(self.D),
            'col': spaces.Discrete(self.D),
            'val': spaces.Discrete(self.D),
        })
        self.max_num_steps = max_num_steps
        self.rewards = {
            'impossible_assign': -1,
            'solved_assign': -1,
            'invalid_assign': -1,
            'backtrack': -1,
            'valid_assign': 0,
            'win': 0,
            'lose': 0,
        }

    def reset(self):
        self.grid_stack = deque()
        self.assign_stack = deque()
        self.num_steps = 0
        self.grid = self._random_puzzle()
        return self.grid

    def step(self, action: Dict):
        """Run one timestep of the environment's dynamics. When end of
        episode is reached, you are responsible for calling `reset()`
        to reset this environment's state.

        Accepts an action and returns a tuple (observation, reward, done, info).

        Args:
            action (object): an action provided by the agent

        Returns:
            observation (object): agent's observation of the current environment
            reward (float) : amount of reward returned after previous action
            done (bool): whether the episode has ended, in which case further step() calls will return undefined results
            info (dict): contains auxiliary diagnostic information (helpful for debugging, and sometimes learning)
        """
        row = action['row']
        col = action['col']
        val = action['val']
        self._check_bounds(row, col, val)
        self.num_steps += 1
        if self.num_steps == self.max_num_steps:
            meta = {'reward': 'lose', 'stack_len': len(self.grid_stack)}
            return self.grid, self.rewards['lose'], True, meta
        # handle invalid actions
        valid_actions = self.grid[row][col].nonzero()[0]
        if len(valid_actions) == 0:
            self.grid = self.grid_stack.pop()
            i,j,k = self.assign_stack.pop()
            self.grid[i][j][k] = 0
            meta = {'reward': 'backtrack', 'stack_len': len(self.grid_stack)}
            return self.grid, self.rewards['backtrack'], False, meta
        if val not in valid_actions:
            meta = {'reward': 'impossible_assign', 'stack_len': len(self.grid_stack)}
            return self.grid, self.rewards['impossible_assign'], False, meta
        if len(valid_actions) == 1:
            meta = {'reward': 'solved_assign', 'stack_len': len(self.grid_stack)}
            return self.grid, self.rewards['solved_assign'], False, meta
        grid = self._assign(self.grid.copy(), row, col, val)
        # handle invalid assign
        if grid is False:
            self.grid[row][col][val] = 0
            meta = {'reward': 'invalid_assign', 'stack_len': len(self.grid_stack)}
            return self.grid, self.rewards['invalid_assign'], False, meta
        self.grid = grid
        # handle solved
        if (self.grid.sum(axis=-1) == 1).all():
            meta = {'reward': 'win', 'stack_len': len(self.grid_stack)}
            return self.grid, self.rewards['win'], True, meta
        # handle valid but not solved
        else:
            self.grid_stack.append(grid.copy()) # save grid state for backtracking (assign may turn out to be invalid)
            self.assign_stack.append((row, col, val)) # for setting assign to invalid after backtracking
            meta = {'reward': 'valid_assign', 'stack_len': len(self.grid_stack)}
            return self.grid, self.rewards['valid_assign'], False, meta

    def render(self, mode='ansi'):
        # TODO: clean up
        s = ""
        width = 1 + max(len(str(v)) for v in range(self.D))
        width = max(width,3)
        line = '+' + '+'.join(['-' * (width * self.d)] * self.d) + '+'
        s += line + '\n'
        for i in range(self.D):
            for j in range(self.D):
                if j == 0:
                    s += '|'
                vals = self.grid[i][j].nonzero()[0]
                if len(vals) == 1:
                    s += str(vals[0]).center(width)
                else:
                    s += '.'.center(width)
                if (j + 1) % self.d == 0:
                    s += '|'
            s += '\n'
            if (i + 1) % self.d == 0:
                s += line + '\n'
        return s

    def seed(self, seed=1337):
        random.seed(seed)
        return [seed]

    def _random_puzzle(self):
        grid = np.ones((self.D, self.D, self.D), dtype=int)
        coords = [(i,j) for i in range(self.D) for j in range(self.D)]
        random.shuffle(coords)
        assigned_cells = 0
        assigned_vals = set()
        for i,j in coords:
            val = random.choice(grid[i][j].nonzero()[0])
            if self._assign(grid, i, j, val) is False:
                return self._random_puzzle()
            assigned_cells += 1
            assigned_vals.add(val)
            if assigned_cells >= self.N and len(assigned_vals) >= self.D - 1:
                return grid
        return self._random_puzzle()

    def _assign(self, grid: np.ndarray, row: int, col: int, val: int):
        other_vals = grid[row][col].nonzero()[0]
        other_vals = other_vals[other_vals != val]
        for ov in other_vals:
            if self._eliminate(grid, row, col, ov) is False:
                return False
        return grid

    def _eliminate(self, grid: np.ndarray, row: int, col: int, val: int):
        vals_binary = grid[row][col] # binary encoding of possible vals
        # check if val already eliminated during constraint propagation
        if vals_binary[val] == 0:
            return grid
        # eliminate val and get remaining possible vals
        vals_binary[val] = 0
        remaining_vals = vals_binary.nonzero()[0]
        # if eliminate last val, contradiction during constraint propagation
        if len(remaining_vals) == 0:
            return False
        # get peers and remove current cell coordinates
        peers = self._get_peers(row, col)
        for peer_coords in peers:
            peer_coords.remove((row, col))
        # strategy 1: reduction to one possible value for a cell
        if len(remaining_vals) == 1:
            for peer_coords in peers:
                for i, j in peer_coords:
                    if self._eliminate(grid, i, j, remaining_vals[0]) is False:
                        return False
        # strategy 2: reduction to one possible cell for a value
        for peer_coords in peers:

            peer_cells = [idx for idx, (i, j) in enumerate(peer_coords) if grid[i][j][val] == 1]
            if len(peer_cells) == 0:
                return False
            if len(peer_cells) == 1:
                i,j = peer_coords[peer_cells[0]]
                if self._assign(grid, i, j, val) is False:
                    return False
        return grid

    def _get_peers(self, row: int, col: int):
        col_unit = [(i, col) for i in range(self.D)]
        row_unit = [(row, j) for j in range(self.D)]
        sq_row = row - row % self.d
        sq_col = col - col % self.d
        sq_unit = [(sq_row + i, sq_col + j) for i in range(self.d) for j in range(self.d)]
        return [col_unit, row_unit, sq_unit]

    def _check_bounds(self, row: int, col: int, val: int):
        assert 0 <= row <= self.D-1, f"row must be in range [0,{self.D-1}]"
        assert 0 <= col <= self.D-1, f"col must be in range [0,{self.D-1}]"
        assert 0 <= val <= self.D-1, f"cell value must be in range [0,{self.D-1}]"


if __name__ == '__main__':
    # TODO: clean this up with baseline agent, pytest, and proper logging
    def get_action(grid: np.ndarray):
        num_valid = [(i, j, n) for i, r in enumerate(grid.sum(axis=-1)) for j, n in enumerate(r) if n > 1]
        row, col, _ = min(num_valid, key=lambda x: x[-1])
        val = grid[row][col].nonzero()[0][0]
        return {'row': row, 'col': col, 'val': val}

    def check_grid(grid: np.ndarray):
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
                check_grid(csdku.grid)
                print(f"Trial {i} reward: {total_reward}")
                print(csdku.render())
                break