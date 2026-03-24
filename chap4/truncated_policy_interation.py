import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from grid_world import GridWorld
import numpy as np

def truncated_iteration(env, gamma, threshold):
    width, height = env.env_size
    values = np.zeros((height, width))
    policy = np.ones((height, width, len(env.action_space))) / len(env.action_space) # 归一化

    while True:
        old_policy = policy.copy()

        # policy evaluation
        for _ in range(1000):
            old_values = values.copy()

            for y in range(height):
                for x in range(width):
                    state = (x, y)
                    new_value = 0.0
  
                    for action_idx, action in enumerate(env.action_space):
                        (next_x, next_y), reward = env._get_next_state_and_reward(state, action)
                        new_value += old_policy[y, x, action_idx] * (reward + gamma * old_values[next_y, next_x])
                    values[y, x] = new_value
       
        # policy improvement
        for y in range(height):
            for x in range(width): 
                state = (x, y)  
                q_values = []
                for action_idx, action in enumerate(env.action_space):
                    (next_x, next_y), reward = env._get_next_state_and_reward(state, action)
                    q_values.append(reward + gamma * values[next_y, next_x])

                action_star = int(np.argmax(q_values))
                # policy update
                policy[y, x, :] = 0.0
                policy[y, x, action_star] = 1.0

        if np.array_equal(policy, old_policy):
            break
    return values, policy
        