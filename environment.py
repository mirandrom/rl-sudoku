import argparse
import gym
from gym import spaces
import numpy as np
import random
from collections import deque
import os

import ray
from ray import tune
from ray.tune import grid_search
from ray.rllib.env import EnvContext
from ray.rllib.policy import Policy
from ray.rllib.models import ModelCatalog
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models.torch.fcnet import FullyConnectedNetwork as TorchFC
from ray.rllib.utils.test_utils import check_learning_achieved
from ray.rllib.utils.framework import try_import_torch
torch, nn = try_import_torch()

from typing import Dict, List, Optional
from ray.rllib.utils.typing import TensorType, Tuple, Union

parser = argparse.ArgumentParser()
parser.add_argument("--run", type=str, default="PPO")
parser.add_argument("--as-test", action="store_true")
parser.add_argument("--stop-iters", type=int, default=50)
parser.add_argument("--stop-timesteps", type=int, default=100000)
parser.add_argument("--stop-reward", type=float, default=0.1)


class ConstraintSudoku(gym.Env):
    # Adapted from http://norvig.com/sudoku.html
    def __init__(self, config: EnvContext):
        d = config.get('d', 3)
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
        self.max_num_steps = config.get('max_num_steps', 1e6)
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
            self._backtrack()
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

    def _backtrack(self):
        """No valid assigns remain, therefore the previous assign was invalid.

        Because the previous assign may have rendered other cell/values invalid,
        we pop and revert to the previous grid state to make these cell/values
        valid again. The previous assign is then set to invalid.
        """
        self.grid = self.grid_stack.pop()
        i, j, k = self.assign_stack.pop()
        self.grid[i][j][k] = 0

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


class HeuristicPolicy(Policy):
    def __init__(self, observation_space, action_space, config):
        Policy.__init__(self, observation_space, action_space, config)
        # example parameter
        self.w = 1.0

    def compute_actions(
            self,
            obs_batch: Union[List[TensorType], TensorType],
            state_batches: Optional[List[TensorType]] = None,
            prev_action_batch: Union[List[TensorType], TensorType] = None,
            prev_reward_batch: Union[List[TensorType], TensorType] = None,
            info_batch: Optional[Dict[str, list]] = None,
            episodes: Optional[List["MultiAgentEpisode"]] = None,
            explore: Optional[bool] = None,
            timestep: Optional[int] = None,
            **kwargs) -> \
            Tuple[TensorType, List[TensorType], Dict[str, TensorType]]:
        # return action batch, RNN states, extra values to include in batch
        actions = []
        for grid in obs_batch:
            num_valid = [(i, j, n) for i, r in enumerate(grid.sum(axis=-1))
                         for j, n in enumerate(r) if n > 1]
            row, col, _ = min(num_valid, key=lambda x: x[-1])
            val = grid[row][col].nonzero()[0][0]
            actions += [{'row': row, 'col': col, 'val': val}]
        return actions, [], {}

    def learn_on_batch(self, samples):
        # implement your learning code here
        return {}  # return stats

    def get_weights(self):
        return {"w": self.w}

    def set_weights(self, weights):
        self.w = weights["w"]


class TorchCustomModel(TorchModelV2, nn.Module):
    """Example of a PyTorch custom model that just delegates to a fc-net."""

    def __init__(self, obs_space, action_space, num_outputs, model_config,
                 name):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs,
                              model_config, name)
        nn.Module.__init__(self)

        self.torch_sub_model = TorchFC(obs_space, action_space, num_outputs,
                                       model_config, name)

    def forward(self, input_dict, state, seq_lens):
        input_dict["obs"] = input_dict["obs"].float()
        fc_out, _ = self.torch_sub_model(input_dict, state, seq_lens)
        return fc_out, []

    def value_function(self):
        return torch.reshape(self.torch_sub_model.value_function(), [-1])


if __name__ == '__main__':

    args = parser.parse_args()
    ray.init()

    # Can also register the env creator function explicitly with:
    # register_env("corridor", lambda config: SimpleCorridor(config))

    ModelCatalog.register_custom_model(
        "my_model", TorchCustomModel)

    config = {
        "env": ConstraintSudoku,  # or "corridor" if registered above
        "env_config": {},
        # Use GPUs iff `RLLIB_NUM_GPUS` env var set to > 0.
        "num_gpus": int(os.environ.get("RLLIB_NUM_GPUS", "0")),
        "model": {
            "custom_model": "my_model",
        },
        "lr": grid_search([1e-2, 1e-4, 1e-6]),  # try different lrs
        "num_workers": 1,  # parallelism
        "framework": "torch"
    }

    stop = {
        "training_iteration": args.stop_iters,
        "timesteps_total": args.stop_timesteps,
        "episode_reward_mean": args.stop_reward,
    }

    results = tune.run(args.run, config=config, stop=stop)

    if args.as_test:
        check_learning_achieved(results, args.stop_reward)
    ray.shutdown()

