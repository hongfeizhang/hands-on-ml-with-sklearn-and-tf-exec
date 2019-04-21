import gym

if __name__ == "__main__":
    env=gym.make("CartPole-v0")
    obs=env.reset()
    #print(obs)

    #env.render()
    img=env.render(mode)