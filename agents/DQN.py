import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import gym
from agents.BaseAgent import BaseAgent
from networks.networks import DQN
from networks.network_bodies import AtariBody, SimpleBody
from utils.ReplayMemory import ExperienceReplayMemory, PrioritizedReplayMemory

from timeit import default_timer as timer

class Model(BaseAgent):
    def __init__(self, static_policy=False, env=None, config=None, log_dir='/tmp/gym'):
        super(Model, self).__init__(config=config, env=env, log_dir=log_dir)
        self.device = config.device

        self.noisy=config.USE_NOISY_NETS
        self.priority_replay=config.USE_PRIORITY_REPLAY

        self.gamma = config.GAMMA
        self.lr = config.LR
        self.target_net_update_freq = config.TARGET_NET_UPDATE_FREQ
        self.experience_replay_size = config.EXP_REPLAY_SIZE
        self.batch_size = config.BATCH_SIZE
        self.learn_start = config.LEARN_START
        self.update_freq = config.UPDATE_FREQ
        self.sigma_init= config.SIGMA_INIT
        self.priority_beta_start = config.PRIORITY_BETA_START
        self.priority_beta_frames = config.PRIORITY_BETA_FRAMES
        self.priority_alpha = config.PRIORITY_ALPHA
    
        self.static_policy = static_policy
        self.num_feats = env.observation_space.shape
        self.num_actions = env.action_space.nvec if isinstance(env.action_space, gym.spaces.MultiDiscrete) else env.action_space.n
        self.env = env
        self.loss_function = nn.SmoothL1Loss()
        self.declare_networks()
            
        self.target_model.load_state_dict(self.model.state_dict())
        self.optimizer = optim.RMSprop(self.model.parameters(), lr=self.lr)
        
        #move to correct device
        self.model = self.model.to(self.device)
        self.target_model.to(self.device)

        if self.static_policy:
            self.model.eval()
            self.target_model.eval()
        else:
            self.model.train()
            self.target_model.train()

        self.update_count = 0

        self.declare_memory()

        self.nsteps = config.N_STEPS
        self.nstep_buffer = []

    def declare_networks(self):
        self.model = DQN(self.num_feats, self.num_actions, noisy=self.noisy, sigma_init=self.sigma_init, body=AtariBody)
        self.target_model = DQN(self.num_feats, self.num_actions, noisy=self.noisy, sigma_init=self.sigma_init, body=AtariBody)

    def declare_memory(self):
        self.memory = ExperienceReplayMemory(self.experience_replay_size) if not self.priority_replay else PrioritizedReplayMemory(self.experience_replay_size, self.priority_alpha, self.priority_beta_start, self.priority_beta_frames)

    def append_to_replay(self, s, a, r, s_):
        self.nstep_buffer.append((s, a, r, s_))

        if(len(self.nstep_buffer)<self.nsteps):
            return
        
        R = sum([self.nstep_buffer[i][2]*(self.gamma**i) for i in range(self.nsteps)])
        state, action, _, _ = self.nstep_buffer.pop(0)

        self.memory.push((state, action, R, s_))

    def prep_minibatch(self):
        # random transition batch is taken from experience replay memory
        transitions, indices, weights = self.memory.sample(self.batch_size)
        
        batch_state, batch_action, batch_reward, batch_next_state = zip(*transitions)
        
        shape = [-1,len(self.env.nodes),self.num_feats[1]]

        batch_state = torch.tensor(batch_state, device=self.device, dtype=torch.float).view(shape)
        batch_action = torch.tensor(batch_action, device=self.device, dtype=torch.long).squeeze().view(-1, len(self.num_actions), 1)
        batch_reward = torch.tensor(batch_reward, device=self.device, dtype=torch.float).squeeze().view(-1, 1)
        
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch_next_state)), device=self.device, dtype=torch.bool)
        try: #sometimes all next states are false
            non_final_next_states = torch.tensor([s for s in batch_next_state if s is not None], device=self.device, dtype=torch.float).view(shape)
            empty_next_state_values = False
        except:
            non_final_next_states = None
            empty_next_state_values = True

        return batch_state, batch_action, batch_reward, non_final_next_states, non_final_mask, empty_next_state_values, indices, weights

    def compute_loss(self, batch_vars, s_): #faster
        batch_state, batch_action, batch_reward, non_final_next_states, non_final_mask, empty_next_state_values, indices, weights = batch_vars

        #estimate
        self.model.sample_noise()
        current_q_values = torch.sum(self.model(batch_state).gather(-1, batch_action), dim=-1)
        #target
        with torch.no_grad():
            max_next_q_values = torch.zeros([self.batch_size, len(self.num_actions)], device=self.device, dtype=torch.float).unsqueeze(dim=2)
            if not empty_next_state_values:
                max_next_action = self.get_max_next_state_action(non_final_next_states)
                self.target_model.sample_noise()
                max_next_q_values[non_final_mask] = self.target_model(non_final_next_states).gather(-1, max_next_action)
            expected_q_values = batch_reward + ((self.gamma**self.nsteps)*torch.sum(max_next_q_values, dim=-1))

        diff = torch.mean(torch.abs((expected_q_values - current_q_values)), dim=-1)
        if self.priority_replay:
            self.memory.update_priorities(indices, diff.detach().squeeze().abs().cpu().numpy().tolist())
            loss = self.MSE(diff).squeeze() * weights
            #loss = self.loss_function(current_q_values,expected_q_values).squeeze() * weights
        else:
            loss = self.MSE(diff)
            #loss = self.loss_function(current_q_values,expected_q_values)
        loss = loss.mean()
        return loss

    def update(self, s, a, r, s_, frame=0):
        if self.static_policy:
            return None

        self.append_to_replay(s, a, r, s_)

        if frame < self.learn_start or frame % self.update_freq != 0:
            return None

        batch_vars = self.prep_minibatch()

        loss = self.compute_loss(batch_vars, s_)

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.model.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

        self.update_target_model()
        self.save_td(loss.item(), frame)
        self.save_sigma_param_magnitudes(frame)

    def get_action(self, s, eps=0.1): #faster
        with torch.no_grad():
            X = torch.tensor([s], device=self.device, dtype=torch.float)
            self.model.sample_noise()
            a = self.model(X).max(-1)[1].view(1, -1)
            #print(a)
            act = np.zeros(a.size(),dtype=np.int32)
            #for n in range(a.size()[0]): # batches
            for i in range(a.size()[1]): # diverters
                if np.random.random() >= eps or self.static_policy or self.noisy:
                    act[0][i] = int(a[0][i].item())
                else:
                    act[0][i] = np.random.randint(0, self.num_actions[i])
            return act
    def update_target_model(self):
        self.update_count+=1
        self.update_count = self.update_count % self.target_net_update_freq
        if self.update_count == 0:
            self.target_model.load_state_dict(self.model.state_dict())

    def get_max_next_state_action(self, next_states):
        return self.target_model(next_states).max(dim=-1)[1].view(-1, len(self.num_actions), 1)

    def finish_nstep(self):
        while len(self.nstep_buffer) > 0:
            R = sum([self.nstep_buffer[i][2]*(self.gamma**i) for i in range(len(self.nstep_buffer))])
            state, action, _, _ = self.nstep_buffer.pop(0)

            self.memory.push((state, action, R, None))

    def reset_hx(self):
        pass
    
    def get_state_dict(self):
        return self.model.state_dict()
    
    def load_model_dict(self, PATH):
        return self.model.load_state_dict(torch.load(PATH))
    
    