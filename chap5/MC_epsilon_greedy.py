import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from grid_world import GridWorld
import numpy as np

def MC_epsilon_greedy(env, gamma, epsilon):
    width, height = env.env_size
    n_actions = len(env.action_space)

    policy = np.ones((height, width, n_actions)) / n_actions
    return_sum = np.zeros((height, width, n_actions))
    return_count = np.zeros((height, width, n_actions))
    q_values = np.zeros((height, width, n_actions))

    while True:
        old_policy = policy.copy()
        
        # 生成经验数据
        state = (np.random.randint(width), np.random.randint(height))
        action_idx = np.random.choice(n_actions)
        action = env.action_space[action_idx]

        episode = []

        for _ in range(10000):
            next_state, reward = env._get_next_state_and_reward(state, action)
            episode.append((state, action_idx, reward))
            # 更新state和action
            state = next_state
            action_idx = np.random.choice(n_actions, p=old_policy[state[1], state[0]])
            action = env.action_space[action_idx]

            if env._is_done(state):
                break
        
        G = 0
        for state, action_idx, reward in reversed(episode):
            G = gamma * G + reward
            x, y = state
            return_sum[y, x, action_idx] += G
            return_count[y, x, action_idx] += 1
            q_values[y, x, action_idx] = return_sum[y, x, action_idx] / return_count[y, x, action_idx]

            action_star = int(np.argmax(q_values[y, x, :]))
            policy[y, x, :] = epsilon / n_actions
            policy[y, x, action_star] = 1 -epsilon + epsilon / n_actions

        if np.array_equal(old_policy, policy):
            break
    return policy