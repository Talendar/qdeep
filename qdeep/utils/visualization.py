import time

import numpy as np
import tensorflow as tf


def visualize_policy(policy,
                     env,
                     num_episodes: int = 1,
                     fps: int = 60,
                     epsilon_greedy: float = 0):
    for episode in range(num_episodes):
        obs = env.reset().observation
        episode_reward = 0.0
        done = False

        while not done:
            env.render(mode="human")
            time.sleep(1 / fps)

            # Random action:
            if np.random.uniform(low=0, high=1) < epsilon_greedy:
                action = np.random.randint(low=0,
                                           high=env.action_spec().num_values)
            # Greedy policy:
            else:
                q_values = policy(tf.expand_dims(obs, axis=0))[0]
                action = tf.argmax(q_values).numpy()

            timestep_obj = env.step(action)
            obs = timestep_obj.observation

            episode_reward += timestep_obj.reward
            done = timestep_obj.last()

        print(f"Episode reward: {episode_reward:.2f}")
