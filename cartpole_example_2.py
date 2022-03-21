import gym

# load up any environment! But let's go with the cartpole just for shits and giggles
env = gym.make('CartPole-v0')
env.reset()

# a new concept arrived! an "episode" one full run of the simulationself.
# this changes depending on the problem and can be defined however we want
# openAI gym defines an episode to end for cartpole when:
# (1) the pole is more than 15 degrees from vertical or
# (2) the cart moves more than 2.4 units from the center.
for i_episode in range(20):
    observation = env.reset()
    # here we define how long the episode will run, say 100 "time-steps"
    for t in range(100):
        env.render()  # we are all visual person, let's see the actual cartpole
        print(observation)
        # actions are move one unit left or one unit right
        action = env.action_space.sample()

        # No more random action!
        # env.step(env.action_space.sample())  # take a random action, no learning yet!
        # remember that `step` is one of the very important functions!
        observation, reward, done, info = env.step(action)
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break
env.close()
