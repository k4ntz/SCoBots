from scobi.core import Environment

env = Environment('PongDeterministic-v4')

aspace = env.action_space
ospace = env.observation_space

print(aspace)
print(len(ospace))

env.reset()
obs, _, _, _, info = env.step(1)
print(obs)