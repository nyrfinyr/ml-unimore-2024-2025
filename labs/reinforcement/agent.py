from typing import Any

import numpy as np
import random

from numpy import signedinteger


class Agent:
  """
  Class that models a reinforcement learning agent.
  """

  def __init__(self, n_rows, n_cols, epsilon=0.01, alpha=1, gamma=1):
    self.n_rows = n_rows
    self.n_cols = n_cols

    self.n_actions = 4

    self.epsilon = epsilon
    self.alpha = alpha
    self.gamma = gamma

    self.Q = np.random.rand(self.n_rows, self.n_cols, self.n_actions)

  def get_action_eps_greedy(self, r:int, c:int) -> int:
    """
    Epsilon-greedy sampling of next action given the current state.
    
    Parameters
    ----------
    r: int
      Current `y` position in the labyrinth
    c: int
      Current `x` position in the labyrinth

    Returns
    -------
    action: int
      Action sampled according to epsilon-greedy policy.
    """
    if random.random() >= self.epsilon:
      return self.get_action_greedy(r, c)
    return random.randint(0, self.n_actions - 1)


  def get_action_greedy(self, r:int, c:int) -> int:
    """
    Greedy sampling of next action given the current state.

    Parameters
    ----------
    r: int
      Current `y` position in the labyrinth
    c: int
      Current `x` position in the labyrinth

    Returns
    -------
    action: int
      Action sampled according to greedy policy.
    """
    return int(np.argmax(self.Q[r, c]))

  def update_Q(self, old_state, action, reward, new_state):
    raise NotImplementedError()
