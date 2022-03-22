import gym

# load up any environment! But let's go with the cartpole just for shits and giggles
env = gym.make('CartPole-v0')
env.reset()

# a new concept arrived! an "episode" one full run of the simulation.
# this changes depending on the problem and can be defined however we want
episodes = 10000
num_steps = 100
log_interval = 10
running_reward = 10

for i_episode in range(episodes):
    # every episode is a new start, or a great start
    observation, episode_reward = env.reset(), 0

    # here we define how long the episode will run, say 100 "time-steps"
    for t in range(num_steps):
        env.render()  # we are all visual people, let's see the actual cartpole
        # let's still take random actions!
        action = env.action_space.sample()
        # remember that `step` is one of the very important functions!
        # when you are doing your custom gym environment, this is the most
        # important function to pay attention to.

        # observation: states of the environment after performing `action`
        #   For the cartpole problem is a 4-tuple:
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
        episode_reward += reward

        if done:
            break

    running_reward = 0.05 * episode_reward + (1 - 0.05) * running_reward

    if i_episode % log_interval == 0:
        print("Observations: {}".format(observation))
        print("Episode {}\tLast reward: {:.2f}\tAverage reward: {:.2f}".format(
            i_episode, episode_reward, running_reward))

env.close()
