import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from grid_world import GridWorld
import numpy as np

def Policy_searching_by_Sarsa(env, gamma, alpha, epsilon):
    width, height = env.env_size
    n_actions = len(env.action_space)

    policy = np.ones((height, width, n_actions)) / n_actions
    q_values = np.zeros((height, width, n_actions))
    
    while True:
        old_policy = policy.copy()
        state = (np.random.randint(width), np.random.randint(height))
        if env._is_done(state):
            continue
        (x, y) = state
        action_idx = np.random.choice(n_actions, p=old_policy[y, x])
        action = env.action_space[action_idx]

        next_state, reward = env._get_next_state_and_reward(state, action)
        (next_x, next_y) = next_state
        if env._is_done(next_state):
            target = reward
        else:
            next_action_idx = np.random.choice(n_actions, p=old_policy[next_y, next_x])
            target = reward + gamma * q_values[next_y, next_x, next_action_idx]
        # update q-value
        q_values[state[1], state[0], action_idx] += alpha * (target - q_values[y, x, action_idx])
        # update policy
        action_star = int(np.argmax(q_values[y, x, :]))
        policy[y, x, :] = epsilon / n_actions
        policy[y, x, action_star] = 1- epsilon + epsilon / n_actions

        if np.array_equal(old_policy, policy):
            break
    return policy