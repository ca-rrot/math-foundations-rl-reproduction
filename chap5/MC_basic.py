import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from grid_world import GridWorld
import numpy as np


def MC_basic(env, gamma):
    width, height = env.env_size
    policy = np.ones((height, width, len(env.action_space))) / len(env.action_space)
    
    while True:
        old_policy = policy.copy()
        
        for y in range(height):
            for x in range(width):

                if env._is_done((x, y)):
                    continue
                
                q_values = []
                for  action in env.action_space:
                   
                    sum_ret = 0
                    # 200 episodes
                    for _ in range(200):
                        ret = 0
                        state = (x, y)
                        # the first episode must start from (s, a)
                        (next_x, next_y), reward = env._get_next_state_and_reward(state, action)
                        ret += reward
                        state = (next_x, next_y)

                        if env._is_done(state):
                            sum_ret += ret
                            continue
                        
                        for t in range(1, 100):
                            # 按照当前policy选择一个action
                            action_idx = np.random.choice(len(env.action_space), p=old_policy[state[1], state[0], :])
                            sample_action = env.action_space[action_idx]

                            (next_x, next_y), reward = env._get_next_state_and_reward(state, sample_action)
                            ret += (gamma**t) * reward
                            state = (next_x, next_y)
                            if env._is_done((next_x, next_y)):
                                break
                        sum_ret += ret
                    q_values.append(sum_ret/200)

                action_star = int(np.argmax(q_values))

                policy[y, x, :] = 0
                policy[y, x, action_star] = 1
    
        if np.array_equal(old_policy, policy):
            break
    return policy