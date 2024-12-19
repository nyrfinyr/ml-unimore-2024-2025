from agent import Agent

class SarsaAgent(Agent):
  """
  Class that models a reinforcement learning agent.
  """

  def update_Q(self, old_state:tuple, action:int, reward:int, new_state:tuple):
    """
    Update action-value function Q
    
    Parameters
    ----------
    old_state: tuple
      Previous state of the Environment
    action: int
      Action performed to go from `old_state` to `new_state`
    reward: int
      Reward got after action `action`
    new_state: tuple
      Next state of the Environment

    Returns
      The action to be executed next
    -------
    None
    """
    # Q(S, A) ← Q(S, A) + α(R + γQ(S′, A′) − Q(S, A))
    new_action = self.get_action_eps_greedy(new_state[0], new_state[1])
    predicted_q = self.Q[new_state[0], new_state[1], new_action]
    current_q = self.Q[old_state[0], old_state[1], action]
    self.Q[old_state[0], old_state[1], action] += self.alpha * (reward + self.gamma * predicted_q - current_q)
    return new_action
