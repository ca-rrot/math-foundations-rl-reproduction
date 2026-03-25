import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from grid_world import GridWorld
import numpy as np

def MC_exporing_starts(env, gamma):
    width, height = env.env_size
    n_actions = len(env.action_space)

    policy = np.ones((height, width, n_actions)) / n_actions
    returns_sum = np.zeros((height, width, n_actions))
    returns_count = np.zeros((height, width, n_actions))
    q_values = np.zeros((height, width, n_actions))

    while True:
        old_policy = policy.copy()

        
        # wpisode generation, 200条episode
        for _ in range(200):
            # 随机(s, a)
            state = (np.random.randint(width), np.random.randint(height))
            action_idx = np.random.choice(n_actions)
            action = env.action_space[action_idx]

            episode = []

            # 单条episode
            for _ in range(100):
                (next_x, next_y), reward = env._get_next_state_and_reward(state, action)
                episode.append((state, action_idx, reward))

                if env._is_done((next_x, next_y)):
                    break
                
                state = (next_x, next_y)
                action_idx = np.random.choice(len(env.action_space), p=old_policy[next_y, next_x])
                action = env.action_space[action_idx]

            G = 0
            visited_pairs = set()

            for state, action, reward in reversed(episode):
                # policy evaluation and policy improvement
                G = reward + gamma * G

                if (state, action_idx) in visited_pairs:
                    continue
                visited_pairs.add((state, action_idx))

                x, y = state
                returns_sum[y, x, action_idx] += G
                returns_count[y, x, action] += 1
                q_values[y, x, action] = returns_sum[y, x, action] / returns_count[y, x, action]

                action_star = int(np.argmax(q_values[y, x, :]))
                policy[y, x, :] = 0
                policy[y, x, action_star] = 1
                
        if np.array_equal(old_policy, policy):
            break
    return policy
                




                    