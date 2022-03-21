import gym

# load up any environment! But let's go with the cartpole just for shits and giggles
env = gym.make('CartPole-v0')
env.reset()

# a new concept arrived! an "episode" one full run of the simulationself.
# this changes depending on the problem and can be defined however we want
EPISODES = 20
for i_episode in range(EPISODES):
    # every episode is a new start, or a great start
    observation = env.reset()
    # here we define how long the episode will run, say 100 "time-steps"
    for t in range(100):
        env.render()  # we are all visual people, let's see the actual cartpole
        print(observation)
        # let's still take random actions!
        action = env.action_space.sample()
        # remember that `step` is one of the very important functions!
        # when you are doing your custom gym environment, this is the most
        # important function to pay attention to.

        # observation: states of the environment after performing `action`
        #   For the cartpoke problem is a 4-tuple:
        #   x position of cart,
        #   x velocity of cart,
        #   angular position of pole,
        #   angular velocity of pole.
        # reward: whether the action was good or bad, useful for the agent to learn
        #   in this environment, every time-step the pole remains straight up,
        #   rewards is +1, -1 other wise.
        #   The entire problem is "solved" when averaged reward is >= 195 over
        #   100 consecutive episodes.
        # done: has the episode finished?
        # info: useful for debinning
        observation, reward, done, info = env.step(action)
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break
env.close()
