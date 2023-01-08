from tqdm import tqdm
import random
import time
from xrl import agents
from xrl.agents import Agent
from xrl.environments import agym
import gymnasium as gym
from xrl.agents.image_extractors import ColorExtractor, IColorExtractor, ZWhatClassifier
from xrl.agents.policies import RandomPolicy
from parsers import parser


def run_agent(agent, env, render=False):
    """
    run the agent in environment provided and return the reward.
    """
    _, tot_return = env.reset(), 0
    observation, reward, done, info = env.step(1)
    if render:
        env.render()
    for t in tqdm(range(3000)):
        feature = agent.image_to_feature(observation, None)
        print(feature)
        action = agent.mf_to_action(None)
        observation, reward, done, info = env.step(action)
        tot_return += reward
        if render:
            env.render()
            time.sleep(0.1)
        if done:
            print(tot_return)
            break


args = parser.parse_args()
# game = ["Carnival", "MsPacman", "Pong", "SpaceInvaders", "Tennis"][1]
game = args.game
game_full_name = game + "Deterministic-v4"
if args.ari:
    env = agym.make(game_full_name)
else:
    env = gym.make(game_full_name)
random.seed(0)
env.seed(0)
# import matplotlib.pyplot as plt
# plt.imshow(env.render(mode="rgb_array"))
# plt.show()
# exit()
if args.interactive:
    ice = IColorExtractor(game=game, load=False)
    agent = Agent(f1=ice, f2=None,
                   f3=RandomPolicy(env.action_space.n))
else:
    agent = Agent(f1=ColorExtractor(game=game, load=False), f2=None,
                   f3=RandomPolicy(env.action_space.n))
    agent.feature_extractor.show_objects = True

print(agent)
run_agent(agent, env, render=args.render)
if args.interactive:
    ice.save()
