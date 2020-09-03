#!/usr/bin/env python
# coding: utf-8

#  # Dueling Deep Q Network

#  ## Imports

# In[1]:


import gym, math, glob, sys
import numpy as np
import datetime
import csv

from timeit import default_timer as timer
from datetime import timedelta
from environments.BHS.environment_v5_0 import Environment
from networks.Models import BHSDuelingDQN, BHS_GCN, BHS_SGN, BHS_GIN, BHS_SAGE, BHS_GAT, BHS_GGNN, BHS_NN, BHS_CG, BHS_PNA, BHS_TEST, BHS_GCN_DQN

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

from IPython.display import clear_output
from matplotlib import pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

from utils.wrappers import *
from agents.DQN import Model as DQN_Agent
from utils.ReplayMemory import ExperienceReplayMemory

from utils.hyperparameters import Config
from utils.plot import plot_all_data


#  ## Hyperparameters

# In[2]:


config = Config()

config.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#epsilon variables
config.epsilon_start = 1.0
config.epsilon_final = 0.01
config.epsilon_decay = 300000
config.epsilon_by_frame = lambda frame_idx: config.epsilon_final + (config.epsilon_start - config.epsilon_final) * math.exp(-1. * frame_idx / config.epsilon_decay)

#misc agent variables
config.GAMMA=0.99
config.LR=1e-4
config.USE_PRIORITY_REPLAY = True

#memory
config.TARGET_NET_UPDATE_FREQ = 1000
config.EXP_REPLAY_SIZE = 50000
config.BATCH_SIZE = 32

#Learning control variables
config.LEARN_START = 100
config.MAX_FRAMES=1000000
config.UPDATE_FREQ = 1

#Nstep controls
config.N_STEPS=1

#data logging parameters
config.ACTION_SELECTION_COUNT_FREQUENCY = 1000


#  ## Agent & Network

# In[3]:


class Model(DQN_Agent):
    def __init__(self, static_policy=False, env=None, config=None, log_dir='tmp/gym/', network ="DQN", downsampled = False):
        super(Model, self).__init__(static_policy, env, config, log_dir=log_dir)

    def declare_networks(self):
        if (network == "DQN"):
            print("Model =", network)
            if (downsampled):
                self.model = BHSDuelingDQN([len(self.env.nodes),self.env.observation_space.shape[1]], self.env.action_space.nvec)
                self.target_model = BHSDuelingDQN([len(self.env.nodes),self.env.observation_space.shape[1]], self.env.action_space.nvec)
            else: 
                self.model = BHSDuelingDQN(self.env.observation_space.shape, self.env.action_space.nvec)
                self.target_model = BHSDuelingDQN(self.env.observation_space.shape, self.env.action_space.nvec)
            
        elif (network == "GCN"):
            print("Model =", network)
            if (downsampled):
                edgelist = self.env.edgelist_down.to(self.device)
                edge_weight = self.env.edge_attr.to(self.device)
                self.model = BHS_GCN([len(self.env.nodes),self.env.observation_space.shape[1]], self.env.action_space.nvec, edgelist, edge_weight)
                self.target_model = BHS_GCN([len(self.env.nodes),self.env.observation_space.shape[1]], self.env.action_space.nvec, edgelist, edge_weight)
            else: 
                edgelist = self.env.edgelist.to(self.device)
                edge_weight = torch.ones([edgelist.shape[1]],dtype=torch.float).to(self.device)
                self.model = BHS_GCN(self.env.observation_space.shape, self.env.action_space.nvec, edgelist, edge_weight)
                self.target_model = BHS_GCN(self.env.observation_space.shape, self.env.action_space.nvec, edgelist, edge_weight)
                
        elif (network == "GAT"):
            print("Model =", network)
            if (downsampled):
                edgelist = self.env.edgelist_down.to(self.device)
                self.model = BHS_GAT([len(self.env.nodes),self.env.observation_space.shape[1]], self.env.action_space.nvec, edgelist)
                self.target_model = BHS_GAT([len(self.env.nodes),self.env.observation_space.shape[1]], self.env.action_space.nvec, edgelist)
            else: 
                edgelist = self.env.edgelist.to(self.device)
                self.model = BHS_GAT(self.env.observation_space.shape, self.env.action_space.nvec, edgelist)
                self.target_model = BHS_GAT(self.env.observation_space.shape, self.env.action_space.nvec, edgelist)
                
        elif (network == "SGN"):
            print("Model =", network)
            if (downsampled):
                edgelist = self.env.edgelist_down.to(self.device)
                edge_weight = self.env.edge_attr.to(self.device)
                self.model = BHS_SGN([len(self.env.nodes),self.env.observation_space.shape[1]], self.env.action_space.nvec, edgelist, edge_weight)
                self.target_model = BHS_SGN([len(self.env.nodes),self.env.observation_space.shape[1]], self.env.action_space.nvec, edgelist, edge_weight)
            else: 
                edgelist = self.env.edgelist.to(self.device)
                edge_weight = torch.ones([edgelist.shape[1]],dtype=torch.float).to(self.device)
                self.model = BHS_SGN(self.env.observation_space.shape, self.env.action_space.nvec, edgelist, edge_weight)
                self.target_model = BHS_SGN(self.env.observation_space.shape, self.env.action_space.nvec, edgelist, edge_weight)
                
        elif (network == "GGNN"):
            print("Model =", network)
            if (downsampled):
                edgelist = self.env.edgelist_down.to(self.device)
                edge_weight = self.env.edge_attr.to(self.device)
                self.model = BHS_GGNN([len(self.env.nodes),self.env.observation_space.shape[1]], self.env.action_space.nvec, edgelist, edge_weight)
                self.target_model = BHS_GGNN([len(self.env.nodes),self.env.observation_space.shape[1]], self.env.action_space.nvec, edgelist, edge_weight)
            else: 
                edgelist = self.env.edgelist.to(self.device)
                edge_weight = torch.ones([edgelist.shape[1]],dtype=torch.float).to(self.device)
                self.model = BHS_GGNN(self.env.observation_space.shape, self.env.action_space.nvec, edgelist, edge_weight)
                self.target_model = BHS_GGNN(self.env.observation_space.shape, self.env.action_space.nvec, edgelist, edge_weight)
            
        elif (network == "SAGE"):
            print("Model =", network)
            if (downsampled):
                graph = self.env.graph_down.to(self.device)
                self.model = BHS_SAGE([len(self.env.nodes),self.env.observation_space.shape[1]], self.env.action_space.nvec, graph)
                self.target_model = BHS_SAGE([len(self.env.nodes),self.env.observation_space.shape[1]], self.env.action_space.nvec, graph)
            else: 
                graph = self.env.graph.to(self.device)
                self.model = BHS_SAGE(self.env.observation_space.shape, self.env.action_space.nvec, graph)
                self.target_model = BHS_SAGE(self.env.observation_space.shape, self.env.action_space.nvec, graph)
            
        elif (network == "GIN"):
            print("Model =", network)
            if (downsampled):
                edgelist = self.env.edgelist_down.to(self.device)
                self.model = BHS_GIN([len(self.env.nodes),self.env.observation_space.shape[1]], self.env.action_space.nvec, edgelist)
                self.target_model = BHS_GIN([len(self.env.nodes),self.env.observation_space.shape[1]], self.env.action_space.nvec, edgelist)
            else: 
                edgelist = self.env.edgelist.to(self.device)
                self.model = BHS_GIN(self.env.observation_space.shape, self.env.action_space.nvec, edgelist)
                self.target_model = BHS_GIN(self.env.observation_space.shape, self.env.action_space.nvec, edgelist)
            
        elif (network == "NN"):
            print("Model =", network)
            if (downsampled):
                edgelist = self.env.edgelist_down.to(self.device)
                edge_attr = self.env.edge_attr.to(self.device)
                edge_attr = edge_attr.view(edge_attr.shape[0],1)
                self.model = BHS_NN([len(self.env.nodes),self.env.observation_space.shape[1]], self.env.action_space.nvec, edgelist, edge_attr)
                self.target_model = BHS_NN([len(self.env.nodes),self.env.observation_space.shape[1]], self.env.action_space.nvec, edgelist, edge_attr)
            else: 
                edgelist = self.env.edgelist.to(self.device)
                edge_attr = torch.ones([edgelist.shape[1],1],dtype=torch.float).to(self.device)
                self.model = BHS_NN(self.env.observation_space.shape, self.env.action_space.nvec, edgelist, edge_attr)
                self.target_model = BHS_NN(self.env.observation_space.shape, self.env.action_space.nvec, edgelist, edge_attr)
                
        elif (network == "CG"):
            print("Model =", network)
            if (downsampled):
                edgelist = self.env.edgelist_down.to(self.device)
                edge_attr = self.env.edge_attr.to(self.device)
                edge_attr = edge_attr.view(edge_attr.shape[0],1)
                self.model = BHS_CG([len(self.env.nodes),self.env.observation_space.shape[1]], self.env.action_space.nvec, edgelist, edge_attr)
                self.target_model = BHS_CG([len(self.env.nodes),self.env.observation_space.shape[1]], self.env.action_space.nvec, edgelist, edge_attr)
            else: 
                edgelist = self.env.edgelist.to(self.device)
                edge_attr = torch.ones([edgelist.shape[1],1],dtype=torch.float).to(self.device)
                self.model = BHS_CG(self.env.observation_space.shape, self.env.action_space.nvec, edgelist, edge_attr)
                self.target_model = BHS_CG(self.env.observation_space.shape, self.env.action_space.nvec, edgelist, edge_attr)
                
        elif (network == "PNA"):
            print("Model =", network)
            if (downsampled):
                edgelist = self.env.edgelist_down.to(self.device)
                edge_attr = self.env.edge_attr.to(self.device)
                edge_attr = edge_attr.view(edge_attr.shape[0],1)
                self.model = BHS_PNA([len(self.env.nodes),self.env.observation_space.shape[1]], self.env.action_space.nvec, edgelist, edge_attr)
                self.target_model = BHS_PNA([len(self.env.nodes),self.env.observation_space.shape[1]], self.env.action_space.nvec, edgelist, edge_attr)
            else: 
                edgelist = self.env.edgelist.to(self.device)
                edge_attr = torch.ones([edgelist.shape[1],1],dtype=torch.float).to(self.device)
                self.model = BHS_PNA(self.env.observation_space.shape, self.env.action_space.nvec, edgelist, edge_attr)
                self.target_model = BHS_PNA(self.env.observation_space.shape, self.env.action_space.nvec, edgelist, edge_attr)
            
        elif (network == "GCN_DQN"):
            print("Model =", network)
            if (downsampled):
                edgelist = self.env.edgelist_down.to(self.device)
                edge_weight = self.env.edge_attr.to(self.device)
                self.model = BHS_GCN_DQN([len(self.env.nodes),self.env.observation_space.shape[1]], self.env.action_space.nvec, edgelist, edge_weight)
                self.target_model = BHS_GCN_DQN([len(self.env.nodes),self.env.observation_space.shape[1]], self.env.action_space.nvec, edgelist, edge_weight)
            else: 
                edgelist = self.env.edgelist.to(self.device)
                edge_weight = torch.ones([edgelist.shape[1]],dtype=torch.float).to(self.device)
                self.model = BHS_GCN_DQN(self.env.observation_space.shape, self.env.action_space.nvec, edgelist, edge_weight)
                self.target_model = BHS_GCN_DQN(self.env.observation_space.shape, self.env.action_space.nvec, edgelist, edge_weight)
                
        elif (network == "TEST"):
            print("Model =", network)
            if (downsampled):
                edgelist = self.env.edgelist_down.to(self.device)
                edge_attr = self.env.edge_attr.to(self.device)
                edge_attr = edge_attr.view(edge_attr.shape[0],1)
                hidden = torch.zeros([1,len(self.env.nodes),128],dtype=torch.float).to(self.device)
                self.model = BHS_TEST([len(self.env.nodes),self.env.observation_space.shape[1]], self.env.action_space.nvec, edgelist, edge_attr, hidden)
                self.target_model = BHS_TEST([len(self.env.nodes),self.env.observation_space.shape[1]], self.env.action_space.nvec, edgelist, edge_attr, hidden)
            else: 
                edgelist = self.env.edgelist.to(self.device)
                edge_attr = torch.ones([edgelist.shape[1],1],dtype=torch.float).to(self.device)
                hidden = torch.zeros([5,self.env.observation_space.shape[0],128],dtype=torch.float).to(self.device)
                self.model = BHS_TEST(self.env.observation_space.shape, self.env.action_space.nvec, edgelist, edge_attr, hidden)
                self.target_model = BHS_TEST(self.env.observation_space.shape, self.env.action_space.nvec, edgelist, edge_attr, hidden)
        
        else:
            raise ValueError("Network not chosen - Choose DQN, GCN, GAT, SGN, GGNN, SAGE, GIN, NN, CG, PNA, GCN_DQN or TEST")
            
            


#  ## Training Loop

# In[4]:


start=timer()

if (get_ipython().__class__.__name__ == "ZMQInteractiveShell"):
    network = "NN"
    downsampled = True
elif (len(sys.argv) > 2):
    network = sys.argv[1]
    downsampled = sys.argv[2]
    if (downsampled == "True"):
        downsampled = True
    elif (downsampled == "False"):
        downsampled = False
    else:
        raise ValueError("Downsampling not chosen - Choose True or False")
else:
    raise ValueError("Network or downsampling not chosen")


time = '{date:%Y-%m-%d-%H}'.format(date=datetime.datetime.now())
print(time)

log_dir = "tmp/" + network + "/" + network + "_" + time + "/"
res_dir = "Results/" + network + "/"
filename = res_dir + network + "_" + time
try:
    os.makedirs(res_dir, exist_ok = True)
    os.makedirs(log_dir, exist_ok = True)
except OSError:
    files = glob.glob(os.path.join(log_dir, '*.monitor.csv'))         + glob.glob(os.path.join(log_dir, '*td.csv'))         + glob.glob(os.path.join(log_dir, '*sig_param_mag.csv'))         + glob.glob(os.path.join(log_dir, '*action_log.csv'))
    for f in files:
        os.remove(f)

class Arg_parser():
    def __init__(self):
        self.max_timesteps = 10000000
        self.envtype = 'env_2_0'
        self.tb_log_name = 'DQN'
        self.steplimit = 200
        self.log_interval = 1000
        self.step_penalty = None
        self.trasum_scale = None
        self.destination_score = None
        self.numtotes = 50
        self.randomize_numtotes = False
        self.RL_diverters = None
        self.downsampled = downsampled
    
args = Arg_parser()

env_id = args.envtype
env    = Environment(args) #make_atari(env_id)
env    = bench.Monitor(env, os.path.join(log_dir, env_id))
# env    = wrap_deepmind(env, episode_life=True, clip_rewards=True, frame_stack=False, scale=True)
# env    = ImageToPyTorch(env)
model  = Model(env=env, config=config, log_dir=log_dir, network=network, downsampled=downsampled)

episode_reward = 0
time_get, time_step, time_update = 0,0,0
observation = env.reset(total = True)

for frame_idx in range(1, config.MAX_FRAMES + 1):
    epsilon = config.epsilon_by_frame(frame_idx)

    start_get=timer()
    action = model.get_action(observation, epsilon)[0]
    time_get += timer()-start_get
    
    start_step=timer()
    prev_observation=observation
    observation, reward, done, _, actual_action = env.step(action)
    observation = None if env.deadlock else observation
    time_step += timer()-start_step
    
    action = actual_action
    model.save_action(action, frame_idx) #log action selection
    
    start_update=timer()
    model.update(prev_observation, action, reward, observation, frame_idx)
    episode_reward += reward
    time_update += timer()-start_update
    
    if frame_idx % args.steplimit*10 == 0:
        total_reset = True
    else:
        total_reset = env.deadlock
    if done:
        model.finish_nstep()
        model.reset_hx()
        
        observation = env.reset(total=total_reset)
        model.save_reward(episode_reward)
        episode_reward = 0
    
    
    if frame_idx % args.log_interval == 0:
        torch.save(model.get_state_dict(), filename + ".pt") #model.save_w()
        try:
            clear_output(True)
            print(frame_idx)
            print(time_get/args.log_interval, time_step/args.log_interval, time_update/args.log_interval)
            with open(filename+'.csv', mode='a',newline='') as time_file:
                time_writer = csv.writer(time_file, delimiter=',')
                time_writer.writerow([time_get/args.log_interval, time_step/args.log_interval, time_update/args.log_interval])
                
            time_get, time_step, time_update = 0,0,0
            plot_all_data(log_dir, env_id, 'BHSDuelingDQN', config.MAX_FRAMES, bin_size=(10, 100, 100, 1), smooth=1, time=timedelta(seconds=int(timer()-start)), save_filename=filename+".svg", ipynb=False)
        except IOError:
            pass

torch.save(model.get_state_dict(), filename + ".pt") #model.save_w()
env.close()
plot_all_data(log_dir, env_id, 'BHSDuelingDQN', config.MAX_FRAMES, bin_size=(10, 100, 100, 1), smooth=1, time=timedelta(seconds=int(timer()-start)), save_filename=filename+".svg", ipynb=False)


# In[ ]:




