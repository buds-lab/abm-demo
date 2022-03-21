import gym

# load up any environment! But let's go with the cartpole just for shits and giggles
env = gym.make('CartPole-v0')
env.reset()

# here we define how long the simulation will run, say 1000 "time-steps"
for _ in range(1000):
    env.render()  # we are all visual people, let's see the actual cartpole
    env.step(env.action_space.sample())  # take a random action, no learning yet!
env.close()
