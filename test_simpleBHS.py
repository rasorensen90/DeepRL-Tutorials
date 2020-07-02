import argparse

import os
#os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID" 
#os.environ["CUDA_VISIBLE_DEVICES"]="2"

import sys
sys.path.append('./lib')
import numpy as np
import csv
import torch, math
from utils import logger
from environments.BHS.environment_v5_0 import Environment
from agents.DQN import Model as DQN_Agent
from networks.Models import BHSDuelingDQN, BHS_GCN, BHS_SGN, BHS_GIN, BHS_SAGE, BHS_GAT, BHS_GGNN, BHS_NN, BHS_TEST
from utils.hyperparameters import Config
import time

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
config.N_STEPS=5

#data logging parameters
config.ACTION_SELECTION_COUNT_FREQUENCY = 1000

class Model(DQN_Agent):
    def __init__(self, static_policy=False, env=None, config=None, log_dir='tmp/gym/', network ="DQN"):
        self.network = network
        super(Model, self).__init__(static_policy, env, config, log_dir=log_dir)

    def declare_networks(self):
        if (self.network == "DQN"):
            print("Model =", self.network)
            self.model = BHSDuelingDQN(self.env.observation_space.shape, self.env.action_space.nvec)
            self.target_model = BHSDuelingDQN(self.env.observation_space.shape, self.env.action_space.nvec)
        elif (self.network == "GCN"):
            print("Model =", self.network)
            edgelist = self.env.edgelist.to(self.device)
            self.model = BHS_GCN(self.env.observation_space.shape, self.env.action_space.nvec, edgelist)
            self.target_model = BHS_GCN(self.env.observation_space.shape, self.env.action_space.nvec, edgelist)
        elif (self.network == "GAT"):
            print("Model =", self.network)
            edgelist = self.env.edgelist.to(self.device)
            self.model = BHS_GAT(self.env.observation_space.shape, self.env.action_space.nvec, edgelist)
            self.target_model = BHS_GAT(self.env.observation_space.shape, self.env.action_space.nvec, edgelist)
        elif (self.network == "SGN"):
            print("Model =", self.network)
            edgelist = self.env.edgelist.to(self.device)
            self.model = BHS_SGN(self.env.observation_space.shape, self.env.action_space.nvec, edgelist)
            self.target_model = BHS_SGN(self.env.observation_space.shape, self.env.action_space.nvec, edgelist)
        elif (self.network == "GGNN"):
            print("Model =", self.network)
            edgelist = self.env.edgelist.to(self.device)
            self.model = BHS_GGNN(self.env.observation_space.shape, self.env.action_space.nvec, edgelist)
            self.target_model = BHS_GGNN(self.env.observation_space.shape, self.env.action_space.nvec, edgelist)
        elif (self.network == "SAGE"):
            print("Model =", self.network)
            graph = self.env.graph.to(self.device)
            self.model = BHS_SAGE(self.env.observation_space.shape, self.env.action_space.nvec, graph)
            self.target_model = BHS_SAGE(self.env.observation_space.shape, self.env.action_space.nvec, graph)
        elif (self.network == "GIN"):
            print("Model =", self.network)
            edgelist = self.env.edgelist.to(self.device)
            self.model = BHS_GIN(self.env.observation_space.shape, self.env.action_space.nvec, edgelist)
            self.target_model = BHS_GIN(self.env.observation_space.shape, self.env.action_space.nvec, edgelist)
        elif (self.network == "NN"):
            print("Model =", self.network)
            edgelist = self.env.edgelist.to(self.device)
            edge_attr = torch.ones([edgelist.shape[1]],dtype=torch.float).to(self.device)
            self.model = BHS_NN(self.env.observation_space.shape, self.env.action_space.nvec, edgelist, edge_attr)
            self.target_model = BHS_NN(self.env.observation_space.shape, self.env.action_space.nvec, edgelist, edge_attr)
        elif (self.network == "TEST"):
            print("Model =", self.network)
            edgelist = self.env.edgelist_down.to(self.device)
            edge_attr = self.env.edge_attr.to(self.device)
            self.model = BHS_TEST([len(self.env.nodes),self.env.observation_space.shape[1]], self.env.action_space.nvec, edgelist, edge_attr)
            self.target_model = BHS_TEST([len(self.env.nodes),self.env.observation_space.shape[1]], self.env.action_space.nvec, edgelist, edge_attr)



def writeToteInfo(toteInfo, writedir, controlMethod, numtotes, iteration):
    fn=['Control method', 'Number of totes', 'ToteID', 'Destination','Steps used', 'Destination reached', 'Total Steps used', 'Iteration', 'StepNumber']

    if not os.path.exists(writedir):
        os.makedirs(writedir)
    for toteID in toteInfo:
        totalStepsUsed = toteInfo[toteID]['TotalSteps']
        
        for i in range(len(toteInfo[toteID]['Destinations'])):
            dst = toteInfo[toteID]['Destinations'][i]
            stepsPerDst = toteInfo[toteID]['StepsPerDst'][i]
            dstReached = toteInfo[toteID]['DstReached'][i]
            stepNumber = toteInfo[toteID]['StepNumber'][i]
            
            with open(writedir+controlMethod+'_'+str(numtotes)+'.csv', 'a', newline='') as csvfile:
                            writer = csv.DictWriter(csvfile, fieldnames=fn)
                            writer.writerow({fn[0]: controlMethod,
                                            fn[1]: numtotes,
                                            fn[2]: toteID,
                                            fn[3]: dst,
                                            fn[4]: stepsPerDst,
                                            fn[5]: dstReached,
                                            fn[6]: totalStepsUsed,
                                            fn[7]: iteration,
                                            fn[8]: stepNumber
                                            })

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def main(args):
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID" 
    os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu
    
    env = Environment(args)
    envsize= len(env.elems)
    log_dir = "Results/" + args.network + "/"
    date_time = args.load_from_model
    model  = Model(env=env, config=config, network=args.network)
    model.load_model_dict(log_dir + args.network + "_" + date_time + ".pt")
    base_directory = log_dir + "test/"
    logdir = base_directory + date_time + "/"
    if not os.path.exists(logdir):
        os.makedirs(logdir)
    
    RL_reward_collection = []
    RL_step_collection = []
    RL_deadlock_collection = []
    SSP_reward_collection = []
    SSP_step_collection = []
    SSP_deadlock_collection = []
    DSP_reward_collection = []
    DSP_step_collection = []
    DSP_deadlock_collection = []
    DSPdla_reward_collection = []
    DSPdla_step_collection = []
    DSPdla_deadlock_collection = []
    fn = ['NumTotes', 'Load', 'RL_mean_step', 'SSP_mean_step', 'DSP_mean_step', 'DSPdla_mean_step', 'RL_var_step', 'SSP_var_step', 'DSP_var_step', 'DSPdla_var_step', 'RL_mean_rew', 'SSP_mean_rew', 'DSP_mean_rew', 'DSPdla_mean_rew', 'RL_var_rew', 'SSP_var_rew', 'DSP_var_rew', 'DSPdla_var_rew', 'RL_deadlocks', 'SSP_deadlocks', 'DSP_deadlocks', 'DSPdla_deadlocks']
    seed=0
    prediction_time_total = []
    for n in range(1, envsize+1):
        prediction_time_n = []
        for i in range(args.iterations):
            print('Load: ',n,'\t Iteration: ',i)
            obs = env.reset(total=True, seed=seed, numtotes=n)
            done = False
            RL_reward = 0
            RL_steps=0
            prediction_time = []
            while not done:
                if not args.no_render:
                    env.render()
                start = time.time()
                action = model.get_action(obs)
                end = time.time()
                prediction_time.append(end-start)
#                obs, RLRew, done, tote_info = env.step(action, shortestPath=False, dynamic=False, dla=False)
                obs, RLRew, done, tote_info, _ = env.step(action)#, shortestPath=False, dynamic=False, dla=True)
                RL_reward += RLRew
                RL_steps += 1
    #        print("Episode reward", episode_rew)
            prediction_time_n.append(np.mean(prediction_time))
            print("Average prediction time: ", prediction_time_n[-1])
            print("RL")
            print("Reward = "+str(RL_reward))
            print("Steps = "+str(RL_steps))
            RL_reward_collection.append(RL_reward)
            RL_step_collection.append(RL_steps)
            if RL_steps < env.steplimit:
                RL_deadlock_collection.append(1)
            else:
                RL_deadlock_collection.append(0)
            
            if (args.detailed_log):
                writeToteInfo(tote_info,logdir+'detailed_log/','RL',n,i)
            
            if (not args.RL_only):
                _ = env.reset(total=True, seed=seed, numtotes=n)
                done = False
                SSP_reward = 0
                SSP_steps = 0
                while not done:
                    _, SSPRew, done, tote_info = env.step(shortestPath=True, dynamic=False,  dla=False)
                    SSP_reward += SSPRew
                    SSP_steps += 1
                print("SSP")
                print("Reward = "+str(SSP_reward))
                print("Steps = "+str(SSP_steps))
                SSP_reward_collection.append(SSP_reward)
                SSP_step_collection.append(SSP_steps)
                if SSP_steps < env.steplimit:
                    SSP_deadlock_collection.append(1)
                else:
                    SSP_deadlock_collection.append(0)
                
                if (args.detailed_log):
                    writeToteInfo(tote_info,logdir+'detailed_log/','SSP',n,i)
    
                _ = env.reset(total=True, seed=seed, numtotes=n)
                done = False
                DSP_reward = 0
                DSP_steps = 0
                while not done:
                    _, DSPRew, done, tote_info = env.step(shortestPath=True, dynamic=True,  dla=False)
                    DSP_reward += DSPRew
                    DSP_steps += 1
                print("DSP")
                print("Reward = "+str(DSP_reward))
                print("Steps = "+str(DSP_steps))
                DSP_reward_collection.append(DSP_reward)
                DSP_step_collection.append(DSP_steps)
                if DSP_steps < env.steplimit:
                    DSP_deadlock_collection.append(1)
                else:
                    DSP_deadlock_collection.append(0)
                
                if (args.detailed_log):
                    writeToteInfo(tote_info,logdir+'detailed_log/','DSP',n,i)
                
                _ = env.reset(total=True, seed=seed, numtotes=n)
                done = False
                DSPdla_reward = 0
                DSPdla_steps = 0
                while not done:
                    _, DSPdlaRew, done, tote_info = env.step(shortestPath=True, dynamic=True, dla=True)
                    DSPdla_reward += DSPdlaRew
                    DSPdla_steps += 1
                print("DSPdla")
                print("Reward = "+str(DSPdla_reward))
                print("Steps = "+str(DSPdla_steps))
                DSPdla_reward_collection.append(DSPdla_reward)
                DSPdla_step_collection.append(DSPdla_steps)
                if DSPdla_steps < env.steplimit:
                    DSPdla_deadlock_collection.append(1)
                else:
                    DSPdla_deadlock_collection.append(0)
                
                if (args.detailed_log):
                    writeToteInfo(tote_info,logdir+'detailed_log/','DSP_DLA',n,i)
            
            print()

    
            if i == args.iterations-1:
                print(base_directory)
                logger.record_tabular("totes", n)
                logger.record_tabular("load [%]", 100*n/envsize)
                logger.record_tabular("mean number of steps", round(np.mean(RL_reward_collection), 1))
                logger.record_tabular("episodes", i+1)
                logger.record_tabular("mean episode reward", round(np.mean(RL_reward_collection), 1))
                if (not args.RL_only):
                    logger.record_tabular("SSP reward", SSP_reward)
                    logger.record_tabular("DSP reward", DSP_reward)
                    logger.record_tabular("DSPdla reward", DSPdla_reward)
                    logger.record_tabular("SSP steps", SSP_steps)
                    logger.record_tabular("DSP steps", DSP_steps)
                    logger.record_tabular("DSPdla steps", DSPdla_steps)
                logger.record_tabular("RL reward", RL_reward)
                logger.record_tabular("RL steps", RL_steps)
                logger.dump_tabular()
    
                RL_mean_step = np.mean(RL_step_collection)
                RL_var_step = np.var(RL_step_collection)
                RL_mean_rew = np.mean(RL_reward_collection)
                RL_var_rew = np.var(RL_reward_collection)
                RL_deadlocks = np.sum(RL_deadlock_collection)
                load = n/envsize
                if (not args.RL_only):
                    SSP_mean_step = np.mean(SSP_step_collection)
                    SSP_var_step = np.var(SSP_step_collection)
                    DSP_mean_step = np.mean(DSP_step_collection)
                    DSP_var_step = np.var(DSP_step_collection)
                    DSPdla_mean_step = np.mean(DSPdla_step_collection)
                    DSPdla_var_step = np.var(DSPdla_step_collection)

                    SSP_mean_rew = np.mean(SSP_reward_collection)
                    SSP_var_rew = np.var(SSP_reward_collection)
                    DSP_mean_rew = np.mean(DSP_reward_collection)
                    DSP_var_rew = np.var(DSP_reward_collection)
                    DSPdla_mean_rew = np.mean(DSPdla_reward_collection)
                    DSPdla_var_rew = np.var(DSPdla_reward_collection)

                    SSP_deadlocks = np.sum(SSP_deadlock_collection)
                    DSP_deadlocks = np.sum(DSP_deadlock_collection)
                    DSPdla_deadlocks = np.sum(DSPdla_deadlock_collection)
    
                if (not args.RL_only):
                    with open(logdir+'NumToteTest_'+date_time+'.csv', 'a', newline='') as csvfile:
                        writer = csv.DictWriter(csvfile, fieldnames=fn)
                        writer.writerow({fn[0]: n, 
                                        fn[1]: load,
                                        fn[2]: RL_mean_step,
                                        fn[3]: SSP_mean_step,
                                        fn[4]: DSP_mean_step,
                                        fn[5]: DSPdla_mean_step,
                                        fn[6]: RL_var_step,
                                        fn[7]: SSP_var_step,
                                        fn[8]: DSP_var_step,
                                        fn[9]: DSPdla_var_step,
                                        fn[10]: RL_mean_rew,
                                        fn[11]: SSP_mean_rew,
                                        fn[12]: DSP_mean_rew,
                                        fn[13]: DSPdla_mean_rew,
                                        fn[14]: RL_var_rew,
                                        fn[15]: SSP_var_rew,
                                        fn[16]: DSP_var_rew,
                                        fn[17]: DSPdla_var_rew,
                                        fn[18]: RL_deadlocks,
                                        fn[19]: SSP_deadlocks,
                                        fn[20]: DSP_deadlocks,
                                        fn[21]: DSPdla_deadlocks
                                        })
                else:
                    with open(logdir+'NumToteTest_'+date_time+'.csv', 'a', newline='') as csvfile:
                        writer = csv.DictWriter(csvfile, fieldnames=fn)
                        writer.writerow({fn[0]: n, 
                                        fn[1]: load,
                                        fn[2]: RL_mean_step,
                                        fn[6]: RL_var_step,
                                        fn[10]: RL_mean_rew,
                                        fn[14]: RL_var_rew,
                                        fn[18]: RL_deadlocks
                                        })
    
                RL_reward_collection = []
                RL_step_collection = []
                RL_deadlock_collection = []
                SSP_reward_collection = []
                SSP_step_collection = []
                SSP_deadlock_collection = []
                DSP_reward_collection = []
                DSP_step_collection = []
                DSP_deadlock_collection = []
                DSPdla_reward_collection = []
                DSPdla_step_collection = []
                DSPdla_deadlock_collection = []
            
            seed += 1
        print("Average prediction time: ", np.mean(prediction_time_n))
        [prediction_time_total.append(p) for p in prediction_time_n]
        
    print("Min prediction time: ", np.min(prediction_time_total))
    print("Max prediction time: ", np.max(prediction_time_total))
    print("Average prediction time: ", np.mean(prediction_time_total))
    print("Variance in prediction time: ", np.var(prediction_time_total))
        
        
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Enjoy trained DQN on BHS")
    parser.add_argument('--envtype', type=str, default='env_2_0')
    parser.add_argument('--steplimit', type=int, default=200)
    parser.add_argument('--iterations', type=int, default=100)
    parser.add_argument('--load_from_model', type=str, default="2020-07-02-15")
    parser.add_argument('--network', type=str, default="GCN")
    parser.add_argument('--numtotes', type=int, default=1)
    parser.add_argument('--RL_only', type=str2bool, default=True)
    parser.add_argument('--detailed_log', type=str2bool, default=True)
    parser.add_argument('--no_render', type=str2bool, default=True)
    parser.add_argument('--randomize_numtotes', type=str2bool, default=False)
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--RL_diverters', type=str, default=None)
    
    args = parser.parse_args()
    main(args)
