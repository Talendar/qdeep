import functools
import os
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import numpy as np
import qdeep
import gym

import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')


ENV_NAME = "BreakoutNoFrameskip-v4"
MAX_EPISODE_LEN = 108_000
NUM_STEPS = 100_000


def make_env():
    env = gym.make(ENV_NAME, full_action_space=True)
    return qdeep.wrappers.wrap_all(env, [
        qdeep.wrappers.GymAtariAdapter,
        functools.partial(
            qdeep.wrappers.AtariWrapper,
            to_float=True,
            max_episode_len=MAX_EPISODE_LEN,
            zero_discount_on_life_loss=True,
        ),
        qdeep.wrappers.SinglePrecisionWrapper,
    ])


if __name__ == "__main__":
    env = make_env()
    env_spec = qdeep.make_env_spec(env)

    network = qdeep.utils.DQNAtariNetwork(env_spec.actions.num_values)
    agent = qdeep.DQNAgent(env_spec, network)

    env_loop = qdeep.EnvironmentLoop(env, agent)
    reward_history = env_loop.run(num_steps=NUM_STEPS, render=True)

    avg_hist = [np.mean(reward_history[i:(i + 50)])
                for i in range(len(reward_history) - 50)]
    plt.plot(list(range(len(avg_hist))), avg_hist)
    plt.show()

    while True:
        opt = input("Enter 'Q' to exit or any other key to visualize the "
                    "trained agent.")
        if opt == "q":
            break

        qdeep.utils.visualize_policy(policy=network, env=env, fps=60)
