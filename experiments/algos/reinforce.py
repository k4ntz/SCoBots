""" Reinforce Algorithm"""
import os
import time
import datetime
from torch.utils.tensorboard import SummaryWriter
from torch.distributions import Categorical
from rtpt import RTPT
from termcolor import colored
from pathlib import Path
import numpy as np
import torch
import pandas as pd
from torch import optim
from scobi import Environment
from experiments.algos import networks
from experiments.utils import normalizer, utils
from tqdm import tqdm
#from remix.rules.ruleset import Ruleset

#ruleset = Ruleset().from_file(str(Path().cwd() / "pong.rules"))

EPS = np.finfo(np.float32).eps.item()
BETA = 1 #entropy regularization coefficient
dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class ExperienceBuffer():
    def __init__(self, size, gamma, reward_shaping=False, entropy_beta=0.0):
        self.observations = []
        self.env_rewards = []
        self.sco_rewards = []
        self.values = []
        self.logprobs = []
        self.entropies = []
        self.scobi_reward_shaping = reward_shaping
        self.returns = []
        self.advantages = []
        self.ptr, self.max_size = 0, size
        self.gamma = gamma
        self.entropy_beta = entropy_beta


    def add(self, observation, reward, value, logprob, entropies):
        if self.ptr > self.max_size:
            print("buffer error")
            exit()
        self.observations.append(observation)
        self.env_rewards.append(reward[0])
        self.sco_rewards.append(reward[1])
        self.values.append(value)
        self.logprobs.append(logprob)
        self.entropies = entropies
        self.ptr += 1


    def finalize(self):
        self.env_rewards = np.array(self.env_rewards)
        self.sco_rewards = np.array(self.sco_rewards, dtype=float)
        self.entropies = np.array(self.entropies)
        ret = 0
        if self.scobi_reward_shaping == 2: # mix rewards
            total_rewards = self.sco_rewards + self.env_rewards
        elif self.scobi_reward_shaping == 1: # scobi only
            total_rewards = self.sco_rewards
        else: 
            total_rewards = self.env_rewards # env only
        total_rewards += BETA * self.entropies #entropy regularization
        for reward in total_rewards[::-1]:
            ret = reward + self.gamma * ret
            self.returns.insert(0, ret)
        self.returns = np.array(self.returns)
        vals = torch.cat(self.values).detach().cpu().numpy()
        self.advantages = self.returns - vals


    def get(self):
        mean = self.advantages.mean()
        std = self.advantages.std()
        self.advantages = (self.advantages - mean) / (std + EPS)
        #print([np.min(self.advantages), np.max(self.advantages), np.average(self.advantages)])
        data = { "obs"  : torch.as_tensor(self.observations, device=dev),
                 "rets" : torch.as_tensor(self.returns, device=dev),
                 "advs" : torch.as_tensor(self.advantages, device=dev), 
                 "logps": torch.cat(self.logprobs),
                 "vals" : torch.cat(self.values)
                                        }
        return data


    def reset(self):
        self.ptr = 0
        self.observations = []
        self.env_rewards = []
        self.sco_rewards = []
        self.values = []
        self.logprobs = []
        self.returns = []
        self.advantages = []

def create_dirs(exp_name):
    checkpoint_path = Path(__file__).parent.parent / "checkpoints" / exp_name
    checkpoint_path.mkdir(exist_ok=True)
    return checkpoint_path

def model_name(training_name, episode=1):
    return Path(training_name + "_e"+ str(episode).zfill(2)+".pth")


def select_action(features, policy, random_tr = -1, n_actions=3):
    #action, rules, scores = ruleset.predict_and_explain(features)
    #print(rules[0][0])
    #print()
    #return int(action), (0,0,0,0), (0,0,0,0)
    feature_tensor = torch.tensor(features, device=dev).unsqueeze(0)
    probs = policy(feature_tensor)
    sampler = Categorical(probs)
    action = sampler.sample()
    # select action when no random action should be selected
    if np.random.random() <= random_tr:
        action_id = np.random.random_integers(0, n_actions - 1)
        action[0] = action_id #set action tensor to new value
    log_prob = sampler.log_prob(action) # logprob of action taken
    return action.item(), log_prob, probs.detach().cpu().numpy()



def train(cfg):
    cfg.exp_name = cfg.exp_name + "-seed" + str(cfg.seed)
    ckp_path = create_dirs(cfg.exp_name)
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    writer = SummaryWriter(os.getcwd() + cfg.logdir + cfg.exp_name)

    # init env to get params for policy net
    env = Environment(cfg.env_name,
                      cfg.seed,
                      interactive=cfg.scobi_interactive,
                      reward=cfg.scobi_reward_shaping,
                      hide_properties=cfg.scobi_hide_properties,
                      focus_dir=cfg.scobi_focus_dir,
                      focus_file=cfg.scobi_focus_file)
    n_actions = env.action_space.n
    env.reset()
    obs, _, _, _, _, _, _ = env.step(1)
    hidden_layer_size = cfg.train.policy_h_size
    act_f = cfg.train.policy_act_f
    if hidden_layer_size == 0:
        hidden_layer_size = int(2/3 * (n_actions + len(obs)))
    print("EXPERIMENT")
    print(">> Selected algorithm: REINFORCE")
    print(">> Experiment name:", cfg.exp_name)
    print(">> Seed:", torch.initial_seed())
    print(">> Random Action probability:", cfg.train.random_action_p)
    print(">> Gamma:", cfg.train.gamma)
    print(">> Entropy Regularization beta:", cfg.train.entropy_beta)
    print(">> Learning rate:", cfg.train.learning_rate)
    print(">> Hidden Layer size:", str(hidden_layer_size))
    print(">> Policy Activation Function:", str(act_f))
    print("ENVIRONMENT")
    print(">> Action space: " + str(env.action_space_description))
    print(">> Observation Vector Length:", len(obs))

    # init fresh networks and optimizers
    policy_net = networks.PolicyNet(len(obs), hidden_layer_size, n_actions, act_f).to(dev)
    value_net = networks.ValueNet(len(obs), hidden_layer_size, 1).to(dev)
    policy_optimizer = optim.Adam(policy_net.parameters(), lr=cfg.train.learning_rate)
    value_optimizer = optim.Adam(value_net.parameters(), lr=cfg.train.learning_rate)
    input_normalizer = normalizer.Normalizer(len(obs), clip_value=cfg.train.input_clip_value)
    i_epoch = 1
   
    # overwrite if checkpoint exists
    pol_checkpoints = [str(x) for x in Path(ckp_path).iterdir() if "pol_" + cfg.exp_name in str(x)]
    if pol_checkpoints:
        pol_path = sorted(pol_checkpoints)[-1]
        print(f"Loading latest policy checkpoint: {pol_path.split('/')[-1]}")
        checkpoint = torch.load(pol_path)
        policy_net.load_state_dict(checkpoint["policy"])
        policy_optimizer.load_state_dict(checkpoint["optimizer"])
        input_normalizer.set_state(checkpoint["normalizer_state"])
        torch.set_rng_state(checkpoint["rng_states"][0])
        np.random.set_state(checkpoint["rng_states"][1])
        obs = checkpoint["last_observation"]
        env.reset()
        i_epoch = checkpoint["episode"]
        val_path = ckp_path / model_name("val_" + cfg.exp_name, episode=i_epoch)
        if val_path.exists():
            print(f"Loading corresponding valuenet checkpoint: {val_path.name}")
            checkpoint = torch.load(str(val_path))
            value_net.load_state_dict(checkpoint["value"])
            value_optimizer.load_state_dict(checkpoint["optimizer"])
        else:
            print(f"Fitting valuenet not found ({val_path.name}). Initializing new one.")
        i_epoch += 1


    print("TRAINING")
    print(">> Epochs:", cfg.train.num_episodes)
    print(">> Steps per Epoch:", cfg.train.steps_per_episode)
    print(">> Logging Interval (Steps):", cfg.train.log_steps)
    print(">> Checkpoint Interval (Epochs):", cfg.train.save_every)
    print(">> Current Epoch:", i_epoch)
    print("Training started...")
    # tfboard logging buffer
    tfb_nr_buffer = 0
    tfb_sr_buffer = 0
    tfb_pnl_buffer = 0
    tfb_vnl_buffer = 0
    tfb_pne_buffer = 0
    tfb_step_buffer = 0
    tfb_policy_updates_counter = 0
    last_stdout_nr_buffer = -1000000
    buffer = ExperienceBuffer(cfg.train.max_steps_per_trajectory, cfg.train.gamma, cfg.scobi_reward_shaping, cfg.train.entropy_beta)

    # save model helper function
    def save_models(training_name, episode):
        pol_model_path = ckp_path / model_name("pol_" + training_name, episode)
        val_model_path = ckp_path / model_name("val_" + training_name, episode)
        torch.save({
                "policy": policy_net.state_dict(),
                "episode": episode,
                "optimizer": policy_optimizer.state_dict(),
                "normalizer_state" : input_normalizer.get_state(),
                "rng_states" : [torch.get_rng_state(), np.random.get_state()],
                "last_observation" : obs
                }, pol_model_path)
        torch.save({
                "value": value_net.state_dict(),
                "episode": episode,
                "optimizer": value_optimizer.state_dict(),
                "normalizer_state" : input_normalizer.get_state(),
                "rng_states" : [torch.get_rng_state(), np.random.get_state()],
                "last_observation" : obs
                }, val_model_path)

    # update model parameters
    def update_models(data):
        obss, rets, advs, = data["obs"], data["rets"], data["advs"]
        logps, vals = data["logps"], data["vals"]

        policy_optimizer.zero_grad()
        policy_loss = (-logps * advs).mean()
        policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(policy_net.parameters(), cfg.train.clip_norm)
        policy_optimizer.step()

        val_iters = cfg.train.value_iters
        for _ in range(val_iters):
            value_optimizer.zero_grad()
            value_loss = ((rets - vals)**2).mean()
            value_loss.backward()
            torch.nn.utils.clip_grad_norm_(policy_net.parameters(), cfg.train.clip_norm)
            value_optimizer.step()
            vals = torch.squeeze(value_net.forward(obss.unsqueeze(0)), -1)
        return policy_loss, value_loss

    # training loop
    rtpt = RTPT(name_initials="SeSz",
                experiment_name=cfg.exp_name,
                max_iterations=cfg.train.num_episodes)
    rtpt.start()
    while i_epoch <= cfg.train.num_episodes:
        stdout_nr_buffer = 0
        stdout_pnl_buffer = 0
        stdout_vnl_buffer = 0
        stdout_pne_buffer = 0
        stdout_step_buffer = 0
        stdout_policy_updates_counter = 0
        i_episode_step = 0
        epoch_s_time = time.perf_counter()
        while i_episode_step < cfg.train.steps_per_episode:
            entropies = []
            ep_env_return = 0
            ep_scobi_return = 0
            i_trajectory_step = 0
            incomplete_traj = False
            while i_trajectory_step < cfg.train.max_steps_per_trajectory:

                # interaction
                obs = input_normalizer.normalize(obs)
                action, log_prob, probs = select_action(obs, policy_net,
                                                        cfg.train.random_action_p,
                                                        n_actions)
                value_net_input = torch.tensor(obs, device=dev).unsqueeze(0)
                value_estimation = torch.squeeze(value_net.forward(value_net_input), -1)
                new_obs, natural_reward, scobi_reward, terminated, truncated, _, _ = env.step(action)

                # collection
                entropy = -np.sum([p*np.log(p) for p in probs])
                buffer.add(obs, (natural_reward, scobi_reward), value_estimation, log_prob, entropy)
                entropies.append(entropy)
                ep_env_return += natural_reward
                ep_scobi_return += scobi_reward
                i_trajectory_step += 1
                i_episode_step += 1
                obs = new_obs

                # tfboard logging
                if i_episode_step % cfg.train.log_steps == 0 and tfb_policy_updates_counter > 0:
                    global_step = (i_epoch - 1) * cfg.train.steps_per_episode + i_episode_step
                    avg_nr = tfb_nr_buffer / tfb_policy_updates_counter
                    avg_sr = tfb_sr_buffer / tfb_policy_updates_counter
                    avg_pnl = tfb_pnl_buffer / tfb_policy_updates_counter
                    avg_vnl = tfb_vnl_buffer / tfb_policy_updates_counter
                    avg_pne = tfb_pne_buffer / tfb_policy_updates_counter
                    avg_step = tfb_step_buffer / tfb_policy_updates_counter
                    writer.add_scalar("rewards/avg_return", avg_nr, global_step)
                    writer.add_scalar("rewards/avg_scobi_return", avg_sr, global_step)
                    writer.add_scalar("loss/avg_policy_net", avg_pnl, global_step)
                    writer.add_scalar("loss/avg_value_net", avg_vnl, global_step)
                    writer.add_scalar("loss/avg_policy_net_entropy", avg_pne, global_step)
                    writer.add_scalar("various/avg_steps", avg_step, global_step)
                    tfb_nr_buffer = 0
                    tfb_sr_buffer = 0
                    tfb_pnl_buffer = 0
                    tfb_vnl_buffer = 0
                    tfb_pne_buffer = 0
                    tfb_step_buffer = 0
                    tfb_policy_updates_counter = 0

                # break conditions
                if terminated or truncated:
                    break
                if i_episode_step == cfg.train.steps_per_episode:
                    incomplete_traj = True
                    break

            buffer.finalize()
            # policy update
            data = buffer.get()
            policy_loss, value_loss = update_models(data)
            buffer.reset()
            env.reset()

            policy_loss = policy_loss.detach()
            value_loss = value_loss.detach()
            ep_entropy = np.mean(entropies)

            if not incomplete_traj:
                tfb_policy_updates_counter += 1
                tfb_nr_buffer += ep_env_return
                tfb_sr_buffer += ep_scobi_return
                tfb_pnl_buffer += policy_loss
                tfb_vnl_buffer += value_loss
                tfb_pne_buffer += ep_entropy
                tfb_step_buffer += i_trajectory_step

                stdout_policy_updates_counter += 1
                stdout_nr_buffer += ep_env_return
                stdout_pnl_buffer += policy_loss
                stdout_vnl_buffer += value_loss
                stdout_pne_buffer += ep_entropy
                stdout_step_buffer += i_trajectory_step


        epoch_duration = time.perf_counter() - epoch_s_time
        # checkpointing
        checkpoint_str = ""
        if i_epoch % cfg.train.save_every == 0:
            save_models(cfg.exp_name, i_epoch)
            checkpoint_str = "✔"

        # episode stats
        pcounter = stdout_policy_updates_counter
        tstamp = datetime.datetime.now()
        time_str = tstamp.strftime("%H:%M:%S")

        avg_return_str = utils.color_me(stdout_nr_buffer / pcounter, last_stdout_nr_buffer)
        epoch_count_str = f"{i_epoch:03d}"
        epoch_str = colored(time_str+" Epoch "+epoch_count_str+" >", "blue")
        pne_out = stdout_pne_buffer / pcounter
        vnl_out = stdout_vnl_buffer / pcounter
        step_out = stdout_step_buffer / pcounter
        print(f"{epoch_str} \
              \tavgReturn: {avg_return_str} \
              \tavgEntropy: {pne_out:.2f} \
              \tavgValueNetLoss: {vnl_out:.2f} \
              \tavgSteps: {step_out:.2f} \
              \tDuration: {epoch_duration:.2f}   {checkpoint_str}")
        last_stdout_nr_buffer = stdout_nr_buffer / pcounter
        i_epoch += 1
        rtpt.step()


def train_kangaroo(cfg):
    cfg.exp_name = cfg.exp_name + "-seed" + str(cfg.seed)
    ckp_path = create_dirs(cfg.exp_name)
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    writer = SummaryWriter(os.getcwd() + cfg.logdir + cfg.exp_name)

    # init env to get params for policy net
    env = Environment(cfg.env_name,
                      cfg.seed,
                      interactive=cfg.scobi_interactive,
                      reward=cfg.scobi_reward_shaping,
                      hide_properties=cfg.scobi_hide_properties,
                      focus_dir=cfg.scobi_focus_dir,
                      focus_file=cfg.scobi_focus_file)
    n_actions = env.action_space.n
    env.reset()
    obs, _, _, _, _, _, _ = env.step(1)
    hidden_layer_size = cfg.train.policy_h_size
    act_f = cfg.train.policy_act_f
    if hidden_layer_size == 0:
        hidden_layer_size = int(2/3 * (n_actions + len(obs)))
    print("EXPERIMENT")
    print(">> Selected algorithm: REINFORCE")
    print(">> Experiment name:", cfg.exp_name)
    print(">> Seed:", torch.initial_seed())
    print(">> Random Action probability:", cfg.train.random_action_p)
    print(">> Gamma:", cfg.train.gamma)
    print(">> Entropy Regularization beta:", cfg.train.entropy_beta)
    print(">> Learning rate:", cfg.train.learning_rate)
    print(">> Hidden Layer size:", str(hidden_layer_size))
    print(">> Policy Activation Function:", str(act_f))
    print("ENVIRONMENT")
    print(">> Action space: " + str(env.action_space_description))
    print(">> Observation Vector Length:", len(obs))

    # init fresh networks and optimizers
    policy_net = networks.PolicyNet(len(obs), hidden_layer_size, n_actions, act_f).to(dev)
    value_net = networks.ValueNet(len(obs), hidden_layer_size, 1).to(dev)
    policy_optimizer = optim.Adam(policy_net.parameters(), lr=cfg.train.learning_rate)
    value_optimizer = optim.Adam(value_net.parameters(), lr=cfg.train.learning_rate)
    input_normalizer = normalizer.Normalizer(len(obs), clip_value=cfg.train.input_clip_value)
    i_epoch = 1
   
    # overwrite if checkpoint exists
    pol_checkpoints = [str(x) for x in Path(ckp_path).iterdir() if "pol_" + cfg.exp_name in str(x)]
    if pol_checkpoints:
        pol_path = sorted(pol_checkpoints)[-1]
        print(f"Loading latest policy checkpoint: {pol_path.split('/')[-1]}")
        checkpoint = torch.load(pol_path)
        policy_net.load_state_dict(checkpoint["policy"])
        policy_optimizer.load_state_dict(checkpoint["optimizer"])
        input_normalizer.set_state(checkpoint["normalizer_state"])
        torch.set_rng_state(checkpoint["rng_states"][0])
        np.random.set_state(checkpoint["rng_states"][1])
        obs = checkpoint["last_observation"]
        env.reset()
        i_epoch = checkpoint["episode"]
        val_path = ckp_path / model_name("val_" + cfg.exp_name, episode=i_epoch)
        if val_path.exists():
            print(f"Loading corresponding valuenet checkpoint: {val_path.name}")
            checkpoint = torch.load(str(val_path))
            value_net.load_state_dict(checkpoint["value"])
            value_optimizer.load_state_dict(checkpoint["optimizer"])
        else:
            print(f"Fitting valuenet not found ({val_path.name}). Initializing new one.")
        i_epoch += 1


    print("TRAINING")
    print(">> Epochs:", cfg.train.num_episodes)
    print(">> Steps per Epoch:", cfg.train.steps_per_episode)
    print(">> Logging Interval (Steps):", cfg.train.log_steps)
    print(">> Checkpoint Interval (Epochs):", cfg.train.save_every)
    print(">> Current Epoch:", i_epoch)
    print("Training started...")
    # tfboard logging buffer
    tfb_nr_buffer = 0
    tfb_sr_buffer = 0
    tfb_yposition_buffer = 0
    tfb_epsilon_buffer = 0 
    tfb_pnl_buffer = 0
    tfb_vnl_buffer = 0
    tfb_pne_buffer = 0
    tfb_step_buffer = 0
    tfb_policy_updates_counter = 0
    last_stdout_nr_buffer = -1000000
    buffer = ExperienceBuffer(cfg.train.max_steps_per_trajectory, cfg.train.gamma, cfg.scobi_reward_shaping, cfg.train.entropy_beta)

    # save model helper function
    def save_models(training_name, episode):
        pol_model_path = ckp_path / model_name("pol_" + training_name, episode)
        val_model_path = ckp_path / model_name("val_" + training_name, episode)
        torch.save({
                "policy": policy_net.state_dict(),
                "episode": episode,
                "optimizer": policy_optimizer.state_dict(),
                "normalizer_state" : input_normalizer.get_state(),
                "rng_states" : [torch.get_rng_state(), np.random.get_state()],
                "last_observation" : obs
                }, pol_model_path)
        torch.save({
                "value": value_net.state_dict(),
                "episode": episode,
                "optimizer": value_optimizer.state_dict(),
                "normalizer_state" : input_normalizer.get_state(),
                "rng_states" : [torch.get_rng_state(), np.random.get_state()],
                "last_observation" : obs
                }, val_model_path)

    # update model parameters
    def update_models(data):
        obss, rets, advs, = data["obs"], data["rets"], data["advs"]
        logps, vals = data["logps"], data["vals"]

        policy_optimizer.zero_grad()
        policy_loss = (-logps * advs).mean()
        policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(policy_net.parameters(), cfg.train.clip_norm)
        policy_optimizer.step()

        val_iters = cfg.train.value_iters
        for _ in range(val_iters):
            value_optimizer.zero_grad()
            value_loss = ((rets - vals)**2).mean()
            value_loss.backward()
            torch.nn.utils.clip_grad_norm_(policy_net.parameters(), cfg.train.clip_norm)
            value_optimizer.step()
            vals = torch.squeeze(value_net.forward(obss.unsqueeze(0)), -1)
        return policy_loss, value_loss


    class Epsilon():
        def __init__(self, initial=0.99, final=cfg.train.random_action_p, nb_steps=10000):
            self._init_val = initial
            self.value = initial
            self._final_val = final
            self._nb_steps = nb_steps
            self.decay = (initial - final)/nb_steps
            assert self.decay > 0

        def step(self):
            if self.value < self._final_val:
                return self._final_val
            self.value -= self.decay
            return self.value

    # training loop
    rtpt = RTPT(name_initials="SeSz",
                experiment_name=cfg.exp_name,
                max_iterations=cfg.train.num_episodes)
    rtpt.start()

    # e-greedy stuff
    fv_strs, fv_backmap = env.feature_vector_description
    y_idx = -1
    for i, f in enumerate(fv_strs):
        feature_name = f[0]
        feature_signature = f[1]
        if "POSITION" in feature_name and "Player1" in feature_signature:
            y_idx = np.where(fv_backmap == i)[0][1]
    if y_idx == -1:
        print("Player1 position not part of feature vector")
        exit()
    y_position_milestones = [114, 66] # [124, 76] are platform y coords -10 each, roo is 24px tall
    y_position_epsilons = [Epsilon(), Epsilon()]

    while i_epoch <= cfg.train.num_episodes:
        stdout_nr_buffer = 0
        stdout_pnl_buffer = 0
        stdout_vnl_buffer = 0
        stdout_pne_buffer = 0
        stdout_step_buffer = 0
        stdout_policy_updates_counter = 0
        i_episode_step = 0
        epoch_s_time = time.perf_counter()
        while i_episode_step < cfg.train.steps_per_episode:
            entropies = []
            ep_env_return = 0
            ep_scobi_return = 0
            ep_y_position = 0
            ep_epsilon = 0
            i_trajectory_step = 0
            incomplete_traj = False
            
            while i_trajectory_step < cfg.train.max_steps_per_trajectory:
                # epsilon selection and stepping
                y_value = obs[y_idx] #get y-value of player
                eps_idx = -1
                for i, v in enumerate(y_position_milestones):
                    if y_value < v:
                        eps_idx = i #get epsilon idx for respective milestone
                if eps_idx == -1:
                    random_action_p = cfg.train.random_action_p # use hyperparameter if no milestone passed
                else:
                    random_action_p = y_position_epsilons[eps_idx].step() # else, step in the respective epsilon
                
                # interaction
                obs = input_normalizer.normalize(obs)
                action, log_prob, probs = select_action(obs, policy_net,
                                                        random_action_p,
                                                        n_actions)
                value_net_input = torch.tensor(obs, device=dev).unsqueeze(0)
                value_estimation = torch.squeeze(value_net.forward(value_net_input), -1)
                new_obs, natural_reward, scobi_reward, terminated, truncated, _, _ = env.step(action)

                # collection
                entropy = -np.sum([p*np.log(p) for p in probs])
                buffer.add(obs, (natural_reward, scobi_reward), value_estimation, log_prob, entropy)
                entropies.append(entropy)
                ep_env_return += natural_reward
                ep_scobi_return += scobi_reward
                ep_y_position += y_value
                ep_epsilon += random_action_p
                i_trajectory_step += 1
                i_episode_step += 1
                obs = new_obs

                # tfboard logging
                if i_episode_step % cfg.train.log_steps == 0 and tfb_policy_updates_counter > 0:
                    global_step = (i_epoch - 1) * cfg.train.steps_per_episode + i_episode_step
                    avg_nr = tfb_nr_buffer / tfb_policy_updates_counter
                    avg_sr = tfb_sr_buffer / tfb_policy_updates_counter
                    avg_yposition = tfb_yposition_buffer / tfb_policy_updates_counter
                    avg_epsilon = tfb_epsilon_buffer / tfb_policy_updates_counter
                    avg_pnl = tfb_pnl_buffer / tfb_policy_updates_counter
                    avg_vnl = tfb_vnl_buffer / tfb_policy_updates_counter
                    avg_pne = tfb_pne_buffer / tfb_policy_updates_counter
                    avg_step = tfb_step_buffer / tfb_policy_updates_counter
                    writer.add_scalar("rewards/avg_return", avg_nr, global_step)
                    writer.add_scalar("rewards/avg_scobi_return", avg_sr, global_step)
                    writer.add_scalar("loss/avg_policy_net", avg_pnl, global_step)
                    writer.add_scalar("loss/avg_value_net", avg_vnl, global_step)
                    writer.add_scalar("loss/avg_policy_net_entropy", avg_pne, global_step)
                    writer.add_scalar("various/avg_steps", avg_step, global_step)
                    writer.add_scalar("various/avg_player_y", avg_yposition, global_step)
                    writer.add_scalar("various/avg_epsilon", avg_epsilon, global_step)
                    tfb_nr_buffer = 0
                    tfb_sr_buffer = 0
                    tfb_yposition_buffer = 0
                    tfb_epsilon_buffer = 0 
                    tfb_pnl_buffer = 0
                    tfb_vnl_buffer = 0
                    tfb_pne_buffer = 0
                    tfb_step_buffer = 0
                    tfb_policy_updates_counter = 0

                # break conditions
                if terminated or truncated:
                    break
                if i_episode_step == cfg.train.steps_per_episode:
                    incomplete_traj = True
                    break

            buffer.finalize()
            # policy update
            data = buffer.get()
            policy_loss, value_loss = update_models(data)
            buffer.reset()
            env.reset()

            policy_loss = policy_loss.detach()
            value_loss = value_loss.detach()
            ep_entropy = np.mean(entropies)

            if not incomplete_traj:
                tfb_policy_updates_counter += 1
                tfb_nr_buffer += ep_env_return
                tfb_sr_buffer += ep_scobi_return
                tfb_yposition_buffer += (ep_y_position / i_trajectory_step)
                tfb_epsilon_buffer += (ep_epsilon / i_trajectory_step)
                tfb_pnl_buffer += policy_loss
                tfb_vnl_buffer += value_loss
                tfb_pne_buffer += ep_entropy
                tfb_step_buffer += i_trajectory_step

                stdout_policy_updates_counter += 1
                stdout_nr_buffer += ep_env_return
                stdout_pnl_buffer += policy_loss
                stdout_vnl_buffer += value_loss
                stdout_pne_buffer += ep_entropy
                stdout_step_buffer += i_trajectory_step


        epoch_duration = time.perf_counter() - epoch_s_time
        # checkpointing
        checkpoint_str = ""
        if i_epoch % cfg.train.save_every == 0:
            save_models(cfg.exp_name, i_epoch)
            checkpoint_str = "✔"

        # episode stats
        pcounter = stdout_policy_updates_counter
        tstamp = datetime.datetime.now()
        time_str = tstamp.strftime("%H:%M:%S")

        avg_return_str = utils.color_me(stdout_nr_buffer / pcounter, last_stdout_nr_buffer)
        epoch_count_str = f"{i_epoch:03d}"
        epoch_str = colored(time_str+" Epoch "+epoch_count_str+" >", "blue")
        pne_out = stdout_pne_buffer / pcounter
        vnl_out = stdout_vnl_buffer / pcounter
        step_out = stdout_step_buffer / pcounter
        print(f"{epoch_str} \
              \tavgReturn: {avg_return_str} \
              \tavgEntropy: {pne_out:.2f} \
              \tavgValueNetLoss: {vnl_out:.2f} \
              \tavgSteps: {step_out:.2f} \
              \tDuration: {epoch_duration:.2f}   {checkpoint_str}")
        last_stdout_nr_buffer = stdout_nr_buffer / pcounter
        i_epoch += 1
        rtpt.step()



def eval_reward_discovery(cfg):
    pickle_path = Path(__file__).parent.parent / "results" / "reward_discovery.pkl"
    csv_path = Path(__file__).parent.parent / "results" / "reward_discovery.csv"
    milestones = [1,3,5,10,15,20,25,30]
    seeds = [0,1,2,3,4]
    max_frames_per_seed = 250000 #250k

    df_header = [str(s) for s in milestones]

    if not pickle_path.exists():
        idx_header = ["env", "seed", "scobi-reward"]
        header =  idx_header + df_header
        df = pd.DataFrame(columns=header)
        df = df.set_index(idx_header)
        df.to_pickle(pickle_path)
    df = pd.read_pickle(pickle_path)
    results = np.zeros( (len(seeds), len(milestones)))
    # init env to get params for policy net
    env = Environment(cfg.env_name,
                      cfg.seed,
                      interactive=cfg.scobi_interactive,
                      reward=cfg.scobi_reward_shaping,
                      hide_properties=cfg.scobi_hide_properties,
                      focus_dir=cfg.scobi_focus_dir,
                      focus_file=cfg.scobi_focus_file)
    n_actions = env.action_space.n
    env.reset()
    obs, _, _, _, _, _, _ = env.step(1)
    act_f = cfg.train.policy_act_f
    hidden_layer_size = cfg.train.policy_h_size
    if hidden_layer_size == 0:
        hidden_layer_size = int(2/3 * (n_actions + len(obs)))
    print("REWARD DISCOVERY")
    print(">> Selected algorithm: REINFORCE")
    print(">> Random Action probability:", cfg.train.random_action_p)
    print(">> Gamma:", cfg.train.gamma)
    print(">> Learning rate:", cfg.train.learning_rate)
    print(">> Hidden Layer size:", str(hidden_layer_size))
    print("ENVIRONMENT")
    print(">> Action space: " + str(env.action_space_description))
    print(">> Observation Vector Length:", len(obs))
    print("EVALUATION")
    print(">> Seeds:", seeds)
    print(">> Max Frames per Seed::", max_frames_per_seed)
    print(">> Acc. Reward Milestones:", milestones)
    print("Evaluation started...")

    for run, seed in enumerate(seeds):
        torch.manual_seed(seed)
        np.random.seed(seed)
        env.reset()
        obs, _, _, _, _, _, _ = env.step(1)
        # init fresh policy and optimizer
        policy_net = networks.PolicyNet(len(obs), hidden_layer_size, n_actions, act_f).to(dev)
        value_net = networks.ValueNet(len(obs), hidden_layer_size, 1).to(dev)
        policy_optimizer = optim.Adam(policy_net.parameters(), lr=cfg.train.learning_rate)
        value_optimizer = optim.Adam(value_net.parameters(), lr=cfg.train.learning_rate)
        input_normalizer = normalizer.Normalizer(len(obs), clip_value=cfg.train.input_clip_value) #
        buffer = ExperienceBuffer(cfg.train.max_steps_per_trajectory, cfg.train.gamma, cfg.scobi_reward_shaping)
        ms_counter = 0
        current_milestones = milestones.copy()
        env_info = cfg.env_name
        seed_str = str(seed)
        reward_info = str(cfg.scobi_reward_shaping)
        def update_models(data):
            obss, rets, advs, = data["obs"], data["rets"], data["advs"]
            logps, vals = data["logps"], data["vals"]
            policy_optimizer.zero_grad()
            policy_loss = (-logps * advs).mean()
            policy_loss.backward()
            torch.nn.utils.clip_grad_norm_(policy_net.parameters(), cfg.train.clip_norm)
            policy_optimizer.step()
            val_iters = cfg.train.value_iters
            for _ in range(val_iters):
                value_optimizer.zero_grad()
                value_loss = ((rets - vals)**2).mean()
                value_loss.backward()
                torch.nn.utils.clip_grad_norm_(policy_net.parameters(), cfg.train.clip_norm)
                value_optimizer.step()
                vals = torch.squeeze(value_net.forward(obss.unsqueeze(0)), -1)
            return policy_loss, value_loss
        
        reward_counter = 0
        pbar = tqdm(total=max_frames_per_seed)
        pbar.set_description(f"Seed: {seed}, Remaining Milestones: {current_milestones}")
        for frame in range(1, max_frames_per_seed+1):
            # interaction
            obs = input_normalizer.normalize(obs)
            action, log_prob, probs = select_action(obs, policy_net,
                                                    cfg.train.random_action_p,
                                                    n_actions)
            value_net_input = torch.tensor(obs, device=dev).unsqueeze(0)
            value_estimation = torch.squeeze(value_net.forward(value_net_input), -1)
            new_obs, natural_reward, scobi_reward, terminated, truncated, _, _ = env.step(action)

            # collection
            entropy = -np.sum([p*np.log(p) for p in probs])
            buffer.add(obs, (natural_reward, scobi_reward), value_estimation, log_prob, entropy)
            obs = new_obs

            if natural_reward != 0:
                buffer.finalize()
                # policy update
                data = buffer.get()
                _, _ = update_models(data)
                buffer.reset()

            if natural_reward > 0:
                reward_counter += 1
            if reward_counter in current_milestones:
                results[run,ms_counter] = frame
                ms_counter += 1
                current_milestones.pop(0)
                pbar.set_description(f"Seed: {seed}, Remaining Milestones: {current_milestones}")
            pbar.update(1)
            if len(current_milestones) == 0:
                break

            # break conditions
            if terminated or truncated:
                env.reset()
        pbar.close()
        df.loc[(env_info, seed_str, reward_info), df_header] = results[run]
    mean = results.mean(axis=0).tolist()
    df.loc[(env_info, "mean", reward_info), df_header] = mean
    df.to_pickle(pickle_path)
    df = df.reset_index()
    print(df)
    df.to_csv(str(csv_path))


# eval function, returns trained model
def eval_load(cfg):
    cfg.exp_name = cfg.exp_name + "-seed" + str(cfg.seed)
    ckp_path = create_dirs(cfg.exp_name)
    print("Experiment name:", cfg.exp_name)
    print("Evaluating Mode")
    print("Seed:", cfg.seed)
    print("Random Action probability:", cfg.train.random_action_p)
    # disable gradients as we will not use them
    torch.set_grad_enabled(False)
    # init env
    env = Environment(cfg.env_name,
                      cfg.seed,
                      interactive=cfg.scobi_interactive,
                      reward=cfg.scobi_reward_shaping,
                      hide_properties=cfg.scobi_hide_properties,
                      focus_dir=cfg.scobi_focus_dir,
                      focus_file=cfg.scobi_focus_file)
    n_actions = env.action_space.n
    env.reset()
    obs, _, _, _, _, _, _ = env.step(1)
    hidden_layer_size = cfg.train.policy_h_size
    act_f = cfg.train.policy_act_f
    if hidden_layer_size == 0:
        hidden_layer_size = int(2/3 * (n_actions + len(obs)))
    print("Make hidden layer in nn:", hidden_layer_size)
    policy_net = networks.PolicyNet(len(obs), hidden_layer_size, n_actions, act_f).to(dev)
    normalizer_state = []
    
    # load latest policy checkpoint if exists
    i_epoch = 0
    pol_checkpoints = [str(x) for x in Path(ckp_path).iterdir() if "pol_" + cfg.exp_name in str(x)]
    if pol_checkpoints:
        pol_path = sorted(pol_checkpoints)[-1]
        print(f"Loading latest policy checkpoint: {pol_path}")
        checkpoint = torch.load(pol_path)
        policy_net.load_state_dict(checkpoint["policy"])
        normalizer_state = checkpoint["normalizer_state"]
        i_epoch = checkpoint["episode"]
        print("Epochs trained:", i_epoch)
    input_normalizer = normalizer.Normalizer(v_size=len(obs),
                                             clip_value=cfg.train.input_clip_value,
                                             stats=normalizer_state)
    policy_net.eval()
    return policy_net, input_normalizer, i_epoch
