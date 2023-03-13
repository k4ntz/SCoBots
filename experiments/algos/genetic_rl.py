# deep neuroevolution as a genetic rl algo
import numpy as np
import os
import copy
import multiprocessing
import sys
import warnings
from os import path
sys.path.append(path.dirname(path.dirname(path.dirname(path.abspath(__file__)))))
import torch
import torch.nn as nn
from torch.distributions import Categorical
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from joblib import Parallel, delayed
from rtpt import RTPT

from scobi import Environment
from . import networks


warnings.filterwarnings("ignore", category=DeprecationWarning) 
PATH_TO_OUTPUTS = os.getcwd() + "/checkpoints/"
if not os.path.exists(PATH_TO_OUTPUTS):
    os.makedirs(PATH_TO_OUTPUTS)

model_name = lambda training_name : PATH_TO_OUTPUTS + training_name + "_model.pth"

# TODO: Fix serialization problem and decomment after,
# look at other genetic_rl.py file for details!

dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def init_weights(m):
    # nn.Conv2d weights are of shape [16, 1, 3, 3] i.e. # number of filters, 1, stride, stride
    # nn.Conv2d bias is of shape [16] i.e. # number of filters
    
    # nn.Linear weights are of shape [32, 24336] i.e. # number of input features, number of output features
    # nn.Linear bias is of shape [32] i.e. # number of output features
    
    if ((type(m) == nn.Linear) | (type(m) == nn.Conv2d)):
        torch.nn.init.xavier_uniform(m.weight)
        m.bias.data.fill_(0.00)


# function to create random agents of given count
def return_random_agents(n_inputs, num_agents, n_actions, cfg):
    agents = []
    hidden_layer_size = int(2/3 * (n_actions + len(n_inputs)))
    # TODO: SeSz: still relevant? latest network definitions didnt use this parameter
    if cfg.train.make_hidden:
        print("Agents have", n_inputs, "input nodes,", cfg.train.policy_h_size, "hidden nodes and", n_actions, "output nodes")
    else:
        print("Linear model, no hidden layer! Policy net has", n_inputs, "input nodes and", n_actions, "output nodes")
    for _ in range(num_agents):
        agent = networks.PolicyNet(n_inputs, hidden_layer_size, n_actions).to(dev)
        for param in agent.parameters():
            param.requires_grad = False
        init_weights(agent)
        agents.append(agent) 
    return agents


# function to select action by given features
def select_action(features, policy, random_tr = -1, n_actions=3):
    sample = np.random.random()
    if sample > random_tr:
        # calculate probabilities of taking each action
        probs = policy(torch.tensor(features).unsqueeze(0).float().to(dev))
        # sample an action from that set of probs
        sampler = Categorical(probs)
        action = sampler.sample().item()
    else:
        action = np.random.random_integers(0, n_actions - 1)
    # return action
    return action


# function to run list of agents in given env
def run_agents(env, agents, cfg):
    reward_agents = []
    for agent in agents:
        agent.eval()
        n_actions = env.action_space.n
        _ = env.reset()
        obs, _, done, done2, info, _ = env.step(1)
        features = obs
        r = 0
        t = 0
        while t < cfg.train.max_steps_per_trajectory:
            #if cfg.train.use_raw_features:
            #    features = np.array(np.array([[0,0] if x==None else x for x in raw_features]).tolist()).flatten()
            action = select_action(features, agent, cfg.train.random_action_p, n_actions)
            obs, reward, done, done2, info, _ = env.step(action)
            features = obs
            r = r + reward
            if done or done2:
                break
            t += 1
        if t == cfg.train.max_steps_per_trajectory:
            r = -25
        reward_agents.append(r)
    return reward_agents


# returns average score of given agent when it runs n times
def return_average_score(agent, runs, cfg):
    score = 0.
    env = Environment(cfg.env_name, interactive=cfg.scobi_interactive, focus_dir=cfg.scobi_focus_dir, focus_file=cfg.scobi_focus_file, silent=True)
    rtpt = RTPT(name_initials='SeSz', experiment_name=cfg.exp_name,
                    max_iterations=runs)
    rtpt.start()
    for i in range(runs):
        score += run_agents(env, [agent], cfg)[0]
        rtpt.step()
    avg_score = score/runs
    return avg_score


# gets avg score of every agent running n runs 
def run_agents_n_times(agents, runs, cfg):
    avg_score = []
    agents = tqdm(agents)
    cpu_cores = min(multiprocessing.cpu_count(), cfg.max_cpu_cores)
    avg_score = Parallel(n_jobs=cpu_cores)(delayed(return_average_score)(agent, runs, cfg) for agent in agents)
    return avg_score


# function to mutate given agent to child agent
def mutate(agent):
    child_agent = copy.deepcopy(agent)
    mutation_power = 0.02 #hyper-parameter, set from https://arxiv.org/pdf/1712.06567.pdf
    for param in child_agent.parameters():
        if(len(param.shape)==4): #weights of Conv2D
            for i0 in range(param.shape[0]):
                for i1 in range(param.shape[1]):
                    for i2 in range(param.shape[2]):
                        for i3 in range(param.shape[3]):
                            param[i0][i1][i2][i3]+= mutation_power * np.random.randn()
        elif(len(param.shape)==2): #weights of linear layer
            for i0 in range(param.shape[0]):
                for i1 in range(param.shape[1]):
                    param[i0][i1]+= mutation_power * np.random.randn()
        elif(len(param.shape)==1): #biases of linear layer or conv layer
            for i0 in range(param.shape[0]):
                param[i0]+=mutation_power * np.random.randn()
    return child_agent


# function to add elite to childs 
def add_elite(agents, sorted_parent_indexes, cfg, elite_index=None, only_consider_top_n=10):
    candidate_elite_index = sorted_parent_indexes[:only_consider_top_n]
    if(elite_index is not None):
        candidate_elite_index = np.append(candidate_elite_index,[elite_index]) 
    top_score = None
    top_elite_index = None
    tqdmcandidate_elite_index = tqdm(candidate_elite_index)
    cpu_cores = min(cfg.max_cpu_cores, max(multiprocessing.cpu_count(), only_consider_top_n))
    # elite runs from config
    elite_runs = cfg.train.elite_n_runs
    scores = Parallel(n_jobs=cpu_cores)(delayed(return_average_score)(agents[i], runs=elite_runs, cfg=cfg) for i in tqdmcandidate_elite_index)
    for i, score in enumerate(scores):
        i = candidate_elite_index[i]
        print("Score for elite i ", i, " is ", score)
        if(top_score is None):
            top_score = score
            top_elite_index = i
        elif(score > top_score):
            top_score = score
            top_elite_index = i
    print("Elite selected with index ",top_elite_index, " and score", top_score)
    child_agent = copy.deepcopy(agents[top_elite_index])
    return child_agent


# function to create and return children from given parent agents
def return_children(agents, sorted_parent_indexes, elite_index, cfg):  
    children_agents = []
    #first take selected parents from sorted_parent_indexes and generate N-1 children
    for i in range(len(agents) - 1):
        selected_agent_index = sorted_parent_indexes[np.random.randint(len(sorted_parent_indexes))]
        children_agents.append(mutate(agents[selected_agent_index]))
    #now add one elite
    elite_child = add_elite(agents, sorted_parent_indexes, cfg, elite_index)
    children_agents.append(elite_child)
    elite_index = len(children_agents) - 1 #it is the last one
    return children_agents, elite_index


# save model helper function
def save_agents(training_name, agents, generation, elite_index):
    if not os.path.exists(PATH_TO_OUTPUTS):
        os.makedirs(PATH_TO_OUTPUTS)
    model_path = model_name(training_name)
    print("Saving {}".format(model_path))
    torch.save({
            'agents': agents,
            'generation': generation,
            'elite_index': elite_index
            }, model_path)


# load agents if exists
def load_agents(model_path):
    print("{} does exist, loading ... ".format(model_path))
    checkpoint = torch.load(model_path)
    agents = checkpoint['agents']
    generation = checkpoint['generation']
    elite_index = None
    try:
        elite_index = checkpoint['elite_index']
    except:
        print("No elite index in save available...")
    return agents, generation, elite_index


# train main function
def train(cfg):
    print('Experiment name:', cfg.exp_name)
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    print('Seed:', torch.initial_seed())
    cfg.exp_name = cfg.exp_name + "-seed" + str(cfg.seed)

    writer = SummaryWriter(os.getcwd() + cfg.logdir + cfg.exp_name)

    generations = cfg.train.num_episodes
    print('Generations:', generations)
    print('Max Steps per Episode:', cfg.train.max_steps_per_trajectory)
    print("Random Action probability:", cfg.train.random_action_p)

    # disable gradients as we will not use them
    torch.set_grad_enabled(False)

    # init env to get actions count and features space
    env = Environment(cfg.env_name, interactive=cfg.scobi_interactive, focus_dir=cfg.scobi_focus_dir, focus_file=cfg.scobi_focus_file, silent=True)
    n_actions = env.action_space.n
    _, ep_reward = env.reset(), 0
    obs, _, _, _, info, _ = env.step(1)
    features = obs
    #if cfg.train.use_raw_features:
    #    features = np.array(np.array([[0,0] if x==None else x for x in raw_features]).tolist()).flatten()
    # initialize N number of agents
    num_agents = 500
    print('Number of agents:', num_agents)
    agents = return_random_agents(len(features), num_agents, n_actions, cfg)
    generation = 0

    # load if exists
    model_path = model_name(cfg.exp_name)
    if os.path.isfile(model_path):
        agents, generation, _ = load_agents(model_path)

    # How many top agents to consider as parents
    top_limit = 20
    print('Number of top agents:', top_limit)

    # runs per generation
    n_gen_runs = cfg.train.n_runs
    print('Number of runs per generation:', n_gen_runs)
    print('Number of runs for elite per generation:', cfg.train.elite_n_runs)

    elite_index = None

    rtpt = RTPT(name_initials='SeSz', experiment_name=cfg.exp_name,
                    max_iterations=generations)
    rtpt.start()
    while generation < generations:
        print("Starting generation", generation)
        # return rewards of agents
        rewards = run_agents_n_times(agents, n_gen_runs, cfg) #return average of 3 runs
 
        # sort by rewards
        # reverses and gives top values (argsort sorts by ascending by default) https://stackoverflow.com/questions/16486252/is-it-possible-to-use-argsort-in-descending-order
        sorted_parent_indexes = np.argsort(rewards)[::-1][:top_limit] 
        print("")
        print("")
        
        top_rewards = []
        for best_parent in sorted_parent_indexes:
            top_rewards.append(rewards[best_parent])
       
        print("Generation ", generation, " | Mean rewards: ", np.mean(rewards), " | Mean of top 5: ",np.mean(top_rewards[:5]))
        #print(rewards)
        print("Top ",top_limit," scores", sorted_parent_indexes)
        print("Rewards for top: ",top_rewards)
        
        # setup an empty list for containing children agents
        children_agents, elite_index = return_children(agents, sorted_parent_indexes, elite_index, cfg)
 
        # kill all agents, and replace them with their children
        agents = children_agents

        #log stuff
        writer.add_scalar('Train/Mean rewards', np.mean(rewards), generation)
        writer.add_scalar('Train/Mean of top 5', np.mean(top_rewards[:5]), generation)
        # save generation
        generation += 1
        save_agents(cfg.exp_name, agents, generation, elite_index)
        # make rtpt step
        rtpt.step()


# function to eval best agent of last generation
def eval_load(cfg):
    print('Experiment name:', cfg.exp_name)
    print('Evaluating Mode')
    torch.manual_seed(cfg.seed)
    print('Seed:', cfg.seed)
    cfg.exp_name = cfg.exp_name + "-seed" + str(cfg.seed)
    print("Random Action probability:", cfg.train.random_action_p)
    # disable gradients as we will not use them
    torch.set_grad_enabled(False)
    # init env
    env = Environment(cfg.env_name, interactive=cfg.scobi_interactive, focus_dir=cfg.scobi_focus_dir, focus_file=cfg.scobi_focus_file)
    n_actions = env.action_space.n
    env.reset()
    obs, _, _, _, info, _ = env.step(1)
    features = obs
   # if cfg.train.use_raw_features:
    #    features = np.array(np.array([[0,0] if x==None else x for x in raw_features]).tolist()).flatten()
    # initialize N number of agents
    num_agents = 500
    print('Number of agents:', num_agents)
    agents = return_random_agents(len(features), num_agents, n_actions, cfg)
    generation = 0
    elite_index = 7 # will be overwritten when agents are loaded

    # load if exists
    model_path = model_name(cfg.exp_name)
    if os.path.isfile(model_path):
        agents, generation, t_elite_index = load_agents(model_path)
        if t_elite_index is not None:
            elite_index = t_elite_index
    print('Generation:', generation)
    print('Selected elite agent:', elite_index)
    elite_agent = agents[elite_index]
    # print nn structure
    hidden_layer_size = int(2/3 * (n_actions + len(features)))
    dummy = networks.PolicyNet(len(features), hidden_layer_size, n_actions).to(dev)
    # because old trained runs does not have make_hidden param
    dummy.load_state_dict(elite_agent.state_dict())
    elite_agent = dummy
    #print("Agent reward:", run_agents(env, agent, [elite_agent], cfg))
    model = elite_agent
    return model, select_action

