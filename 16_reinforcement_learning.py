import gym
import matplotlib.pyplot as plt

import tensorflow as tf

import numpy as np
import os
import sys

PROJECT_ROOT_DIR = "."
CHAPTER_ID = "rl"

# to make this notebook's output stable across runs
def reset_graph(seed=42):
    tf.reset_default_graph()
    tf.set_random_seed(seed)
    np.random.seed(seed)

def save_fig(fig_id, tight_layout=True):
    path = os.path.join(PROJECT_ROOT_DIR, "images", CHAPTER_ID, fig_id + ".png")
    print("Saving figure", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format='png', dpi=300)

if __name__ == "__main__":
    env = gym.make('SpaceInvaders-v0')
    env.reset()
    for _ in range(10000):
        env.step(env.action_space.sample())
        env.render('human')
    env.close()