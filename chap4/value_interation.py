import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from grid_world import GridWorld
import numpy as np

def value_interation(env, gamma, threshold):
    width, height = env.env_size
    values = np.zeros((height, width))
    policy = np.zeros((height, width, len(env.action_space)))

    while True:
        old_values = values.copy()
        delta = 0.0

        for y in range(height):
            for x in range(width):
                state = (x, y)
                q_values = []
                for action in env.action_space:
                    (next_x, next_y), reward = env._get_next_state_and_reward(state, action)
                    q_values.append(reward + gamma * old_values[next_y, next_x])

                action_star = int(np.argmax(q_values))
                # 策略更新
                policy[y, x, :] = 0.0
                policy[y, x, action_star] = 1.0
                # 状态值更新
                values[y, x] = q_values[action_star]

                delta = max(delta, abs(values[y, x] - old_values[y, x]))

        if delta < threshold:
            break
    return values, policy

if __name__ == "__main__":
    env = GridWorld()
    env.reset()

    values, policy = value_interation(env, gamma=0.9, threshold=1e-6)
    print(values)

    state, _ = env.reset()
    for t in range(100):
        env.render(animation_interval=0.5)

        x, y = state
        action_idx = int(np.argmax(policy[y, x]))
        action = env.action_space[action_idx]

        next_state, reward, done, _ = env.step(action)
        print(f"Step: {t}, Action: {action}, State: {next_state}, Reward: {reward}, Done: {done}")

        state = next_state
        if done:
            env.render(animation_interval=2)
            break
