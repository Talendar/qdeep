import gym
import qdeep
import tensorflow as tf
import numpy as np

import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')


if __name__ == "__main__":
    env = qdeep.wrappers.SinglePrecisionWrapper(
        qdeep.wrappers.GymWrapper(gym.make("CartPole-v0"))
    )
    env_spec = qdeep.make_env_spec(env)

    network = tf.keras.Sequential([
        tf.keras.layers.Dense(32, activation="relu"),
        tf.keras.layers.Dense(env_spec.actions.num_values, activation="linear"),
    ])

    agent = qdeep.DQNAgent(environment_spec=env_spec,
                           network=network)

    env_loop = qdeep.EnvironmentLoop(environment=env, actor=agent)
    reward_history = env_loop.run(num_steps=int(2e4), render=True)

    avg_hist = [np.mean(reward_history[i:(i + 50)])
                for i in range(len(reward_history) - 50)]
    plt.plot(list(range(len(avg_hist))), avg_hist)
    plt.show()

    qdeep.utils.visualize_policy(policy=network, env=env, fps=60)
