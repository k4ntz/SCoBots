""" Reinforce Algorithm"""
import os
import time
import datetime
from torch.utils.tensorboard import SummaryWriter
from torch.distributions import Categorical
from rtpt import RTPT
from termcolor import colored
import numpy as np
import torch
from torch import optim
from scobi import Environment
from experiments.algos import networks
from experiments.utils import normalizer, utils

EPS = np.finfo(np.float32).eps.item()
PATH_TO_OUTPUTS = os.getcwd() + "/checkpoints/"
if not os.path.exists(PATH_TO_OUTPUTS):
    os.makedirs(PATH_TO_OUTPUTS)

dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class ExperienceBuffer():
    def __init__(self, size, gamma, reward_shaping=False):
        self.observations = []
        self.env_rewards = []
        self.sco_rewards = []
        self.values = []
        self.logprobs = []
        self.scobi_reward_shaping = reward_shaping
        self.returns = []
        self.advantages = []
        self.ptr, self.max_size = 0, size
        self.gamma = gamma


    def add(self, observation, reward, value, logprob):
        if self.ptr > self.max_size:
            print("buffer error")
            exit()
        self.observations.append(observation)
        self.env_rewards.append(reward[0])
        self.sco_rewards.append(reward[1])
        self.values.append(value)
        self.logprobs.append(logprob)
        self.ptr += 1


    def finalize(self):
        self.env_rewards = np.array(self.env_rewards)
        self.sco_rewards = np.array(self.sco_rewards, dtype=float)
        ret = 0
        if self.scobi_reward_shaping:
            total_rewards =  self.sco_rewards 
        else: 
            total_rewards = self.env_rewards
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


def model_name(training_name):
    return PATH_TO_OUTPUTS + training_name + "_model.pth"


def select_action(features, policy, random_tr = -1, n_actions=3):
    feature_tensor = torch.tensor(features, device=dev).unsqueeze(0)
    probs = policy(feature_tensor)
    sampler = Categorical(probs)
    action = sampler.sample()
    log_prob = sampler.log_prob(action)
    # select action when no random action should be selected
    if np.random.random() <= random_tr:
        action = np.random.random_integers(0, n_actions - 1)
    else:
        action = action.item()
    # return action and log prob
    return action, log_prob, probs.detach().cpu().numpy()



def train(cfg):
    cfg.exp_name = cfg.exp_name + "-seed" + str(cfg.seed)
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    writer = SummaryWriter(os.getcwd() + cfg.logdir + cfg.exp_name)

    # init env to get params for policy net
    env = Environment(cfg.env_name,
                      interactive=cfg.scobi_interactive,
                      reward=cfg.scobi_reward_shaping,
                      hide_properties=cfg.scobi_hide_properties,
                      focus_dir=cfg.scobi_focus_dir,
                      focus_file=cfg.scobi_focus_file)
    n_actions = env.action_space.n
    env.reset()
    obs, _, _, _, _, _, _ = env.step(1)
    hidden_layer_size = cfg.train.policy_h_size
    if hidden_layer_size == 0:
        hidden_layer_size = int(2/3 * (n_actions + len(obs)))
    print("EXPERIMENT")
    print(">> Selected algorithm: REINFORCE")
    print(">> Experiment name:", cfg.exp_name)
    print(">> Seed:", torch.initial_seed())
    print(">> Random Action probability:", cfg.train.random_action_p)
    print(">> Gamma:", cfg.train.gamma)
    print(">> Learning rate:", cfg.train.learning_rate)
    print(">> Hidden Layer size:", str(hidden_layer_size))
    print("ENVIRONMENT")
    print(">> Action space: " + str(env.action_space_description))
    print(">> Observation Vector Length:", len(obs))

    # init fresh policy and optimizer

    policy_net = networks.PolicyNet(len(obs), hidden_layer_size, n_actions).to(dev)
    value_net = networks.ValueNet(len(obs), hidden_layer_size, 1).to(dev)
    policy_optimizer = optim.Adam(policy_net.parameters(), lr=cfg.train.learning_rate)
    value_optimizer = optim.Adam(value_net.parameters(), lr=cfg.train.learning_rate)
    input_normalizer = normalizer.Normalizer(len(obs), clip_value=cfg.train.input_clip_value) #
    i_epoch = 1
    # overwrite if checkpoint exists
    model_path = model_name("val_" + cfg.exp_name) # load value net
    if os.path.isfile(model_path):
        print(f"{model_path} does exist, loading ... ")
        checkpoint = torch.load(model_path)
        value_net.load_state_dict(checkpoint["value"])
        value_optimizer.load_state_dict(checkpoint["optimizer"])
        i_epoch = checkpoint["episode"]

    model_path = model_name("pol_" + cfg.exp_name) # load policy net
    if os.path.isfile(model_path):
        print(f"{model_path} does exist, loading ... ")
        checkpoint = torch.load(model_path)
        policy_net.load_state_dict(checkpoint["policy"])
        policy_optimizer.load_state_dict(checkpoint["optimizer"])
        input_normalizer.set_state(checkpoint["normalizer_state"])
        i_epoch = checkpoint["episode"]
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
    buffer = ExperienceBuffer(cfg.train.max_steps_per_trajectory, cfg.train.gamma, cfg.scobi_reward_shaping)

   # def entropy(n_actions, probs):
   #     print(probs)
   #     out = []
   #     for prob in probs:
   #         if prob[0] != 0:
   #             out.append(prob * (np.log(prob) / np.log(n_actions)))
   #         else:
   #             out.append(0)
   #     return(-np.sum(out))
    
    # save model helper function
    def save_models(training_name, episode):
        if not os.path.exists(PATH_TO_OUTPUTS):
            os.makedirs(PATH_TO_OUTPUTS)
        pol_model_path = model_name("pol_" + training_name)
        val_model_path = model_name("val_" + training_name)

        #print("Saving {}".format(model_path))
        torch.save({
                "policy": policy_net.state_dict(),
                "episode": episode,
                "optimizer": policy_optimizer.state_dict(),
                "normalizer_state" : input_normalizer.get_state() 
                }, pol_model_path)
        torch.save({
                "value": value_net.state_dict(),
                "episode": episode,
                "optimizer": value_optimizer.state_dict(),
                "normalizer_state" : input_normalizer.get_state()
                }, val_model_path)

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
                entropy = -np.sum(list(map(lambda p : p * (np.log(p) / np.log(n_actions)) if p[0] != 0 else 0, probs)))
                buffer.add(obs, (natural_reward, scobi_reward), value_estimation, log_prob)
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


# eval function, returns trained model
def eval_load(cfg):
    print("Experiment name:", cfg.exp_name)
    print("Evaluating Mode")
    print("Seed:", cfg.seed)
    print("Random Action probability:", cfg.train.random_action_p)
    # disable gradients as we will not use them
    torch.set_grad_enabled(False)
    # init env
    env = Environment(cfg.env_name,
                      interactive=cfg.scobi_interactive,
                      reward=cfg.scobi_reward_shaping,
                      hide_properties=cfg.scobi_hide_properties,
                      focus_dir=cfg.scobi_focus_dir,
                      focus_file=cfg.scobi_focus_file)
    n_actions = env.action_space.n
    env.reset()
    obs, _, _, _, _, _, _ = env.step(1)
    hidden_layer_size = cfg.train.policy_h_size
    if hidden_layer_size == 0:
        hidden_layer_size = int(2/3 * (n_actions + len(obs)))
    print("Make hidden layer in nn:", hidden_layer_size)
    policy_net = networks.PolicyNet(len(obs), hidden_layer_size, n_actions).to(dev)
    # load if exists
    model_path = model_name("pol_" + cfg.exp_name + "-seed" + str(cfg.seed))
    normalizer_state = []
    if os.path.isfile(model_path):
        print(f"{model_path} does exist, loading ... ")
        checkpoint = torch.load(model_path)
        policy_net.load_state_dict(checkpoint["policy"])
        normalizer_state = checkpoint["normalizer_state"]
        i_epoch = checkpoint["episode"]
        print("Epochs trained:", i_epoch)
    input_normalizer = normalizer.Normalizer(v_size=len(obs),
                                             clip_value=cfg.train.input_clip_value,
                                             stats=normalizer_state)
    policy_net.eval()
    return policy_net, input_normalizer
