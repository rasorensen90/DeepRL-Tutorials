# -*- coding: utf-8 -*-
"""
Created on Wed May 29 09:37:43 2019

@author: RTS
"""
import random
import numpy as np
from .envfactory_v4_1 import env_0_0, env_1_0, env_2_0, env_3_0
from .Element import Element, Diverter, Merger, Toploader
from .Tote import Tote
import gym
from gym import spaces
from .SPGraph import SPGraph, dijsktra

class Environment(gym.Env):
    def __init__(self, args):
        
        self.elems, self.dst, self.src, self.graph, self.GCNMat = globals()[args.envtype]()
        self.totes = []
        self.reward = 0
        self.action_space = []
        self.observation_space = []
        self.numObsFeatures = 1
        self.stepnumber = 0
        self.steplimit = args.steplimit
        self.done = True
        self.default_rand = random.Random(0)
        self.rand_dst = random.Random(0)
        self.rand_src = random.Random(0)
        self.rand_numtotes = random.Random(0)
        self.randomize_numtotes = args.randomize_numtotes
        self.numtotes = args.numtotes
        self.shortestPathTable = self.calcShortestPathTable()
        self.congestion = False
        self.congestion_counter = 0
        self.tote_info = {}
        self.deadlock = False
        
        self.setSpaces()
        
    def getObs(self):
        obs = []
        for e in self.elems:
            if e.tote == None:
                obs.append([0])
            else:
                # obs.append([e.tote.dst])
                if e.tote.dst in self.src:
                    obs.append([-(self.src.index(e.tote.dst)+1)])
                else:
                    obs.append([self.dst.index(e.tote.dst)+1])
        return np.array(obs)
    
    def setDestination(self,tote):
        # TODO - auto detect destinations
        if tote.dst in self.src:
            tote.dst = self.rand_dst.choice(self.dst)#randint(1,len(self.elems)-1)
        else:
            tote.dst = self.rand_src.choice(self.src)
        if tote.ID not in self.tote_info:
            self.tote_info[tote.ID] = {'Tote': tote, 'TotalSteps': 0, 'Destinations': [], 'StepsPerDst': [], 'DstReached': [], 'StepNumber': []}
        self.tote_info[tote.ID]['Destinations'].append(tote.dst)
        self.tote_info[tote.ID]['StepNumber'].append(self.stepnumber)
        self.tote_info[tote.ID]['StepsPerDst'].append(0)
        self.tote_info[tote.ID]['DstReached'].append(0)
    
    def addTotes(self, numtotes):
        for i in range(numtotes):
            self.totes.append(Tote(i,dst=0))
            tote = self.totes[-1]
            self.tote_info[tote.ID] = {'Tote': tote, 'TotalSteps': 0, 'Destinations': [], 'StepsPerDst': [], 'DstReached': [], 'StepNumber': []}
            self.setDestination(tote)
            
            
        e_src = [e for e in self.elems if isinstance(e, Toploader)]
        for t in self.totes:
            src_ = self.rand_src.choice(e_src)
            src_.push(t)
    
    def setSpaces(self):
        # action space
        a = np.ones(len([e for e in self.elems if isinstance(e, Diverter)]),dtype=np.int32)*2 # always two actions (This may change later)
       
        self.action_space = spaces.MultiDiscrete(a)
        
        # observation space
        obs_size = len(self.elems)
        # for p in itertools.chain(self.place, self.placetransbuf):
        #     obs_size.append(1)
        self.observation_space = spaces.Box(low=0, high=1000, shape=(obs_size, self.numObsFeatures),dtype=np.float32)
        self.obs = np.zeros(self.observation_space.shape) 
#        print(self.obs.shape)
        
    def checkForCongestion(self, dla=False, maxCongestion=1.0):
        current_congestion = 0
        for e in [e_ for e_ in self.elems if isinstance(e_, (Merger, Diverter, Toploader))]:
            dla_block = [0,0] # dla only works for diverters with two output directions
            for control in range(len(e.outputElements)):
                if not self.checkEdgeAvailability(e, control, maxCongestion):
                    e.cost[control] = 100
                    dla_block[control] = 1
                    self.congestion = True
                    current_congestion += 1
                else:
                    e.cost[control] = 1
                    dla_block[control] = 0
            if dla and isinstance(e, Diverter):
                if dla_block[0]:
                    e.forced_control = 1
                elif dla_block[1]:
                    e.forced_control = 0
                else: 
                    e.forced_control = None
                
        if self.congestion: # Can be from current step or previous step
            # update shortest path
            self.shortestPathTable = self.calcShortestPathTable()
            if current_congestion == 0: # If solved congestion 
                self.congestion=False
    
    def performDLA(self, action, maxCongestion):
        diverters = [e_ for e_ in self.elems if isinstance(e_, Diverter)]
        for e in diverters:
            dla_block = [0,0] # dla only works for diverters with two output directions
            for control in range(len(e.outputElements)):
                if not self.checkEdgeAvailability(e, control, maxCongestion):
                    dla_block[control] = 1
                else:
                    dla_block[control] = 0

            if dla_block[0]:
                action[diverters.index(e)] = 1
            elif dla_block[1]:
                action[diverters.index(e)] = 0
          
        return action
    
    def recursive_occupency(self, element, control):
        num_elements = 0
        num_occupied = 0
        
        if element.tote is not None and not element.tote.moved:
            num_elements += 1
            num_occupied += 1
        else:
            num_elements += 1
        
        if isinstance(element.outputElements[control], (Merger, Diverter, Toploader)):
            if element.outputElements[control].tote is not None and not element.outputElements[control].tote.moved:
                num_elements += 1
                num_occupied += 1
            else:
                num_elements += 1
            
            return num_elements, num_occupied
        
        else:
            num_elem, num_occ = self.recursive_occupency(element.outputElements[control], 0)
            num_elements += num_elem
            num_occupied += num_occ
            
            return num_elements, num_occupied
    
    def checkEdgeAvailability(self, element, control, maxCongestion = 1.0):
        num_elements, num_occupied = self.recursive_occupency(element,control)
        if num_occupied/num_elements < maxCongestion:   # Available
            return True
        else:                                           # Congested
            return False
        
        
    def calcShortestPathTable(self):        
        graph = SPGraph()
        [graph.add_node(e.ID) for e in self.elems]
        
        [[graph.add_edge(e.ID, e_out.ID, e.cost[e.outputElements.index(e_out)]) for e_out in e.outputElements] for e in self.elems]
        
        shortestPathTable = np.zeros([len([e for e in self.elems if isinstance(e, Diverter)]),len(self.src+self.dst)])

        for n_src in [e.ID for e in self.elems if isinstance(e, Diverter)]:
            _,path = dijsktra(graph,n_src)
            for n_dst in [e.ID for e in self.elems if e.ID in self.src or e.ID in self.dst]:
                k=n_dst
                k_new = k
                while k_new != n_src:
                    k_new = path.get(k)
                    
                    if k_new != n_src:
                        k = k_new
                    if k == n_dst or k == None:
                        break
                ctrl = 0
                for e in [e for e in self.elems if e.ID == n_src]:
                    for e_out in[e_out for e_out in e.outputElements if e_out.ID == k]:
                        ctrl = e.outputElements.index(e_out)
                        eID_src = [e.ID for e in self.elems if isinstance(e, Diverter)].index(n_src)
                        eID_dst = [e.ID for e in self.elems if e.ID in self.src or e.ID in self.dst].index(n_dst)
                        shortestPathTable[eID_src,eID_dst] = ctrl
        return shortestPathTable
    
    def calcShortestPath(self, dynamic=False, dla=False, maxCongestion=1.0):
        if (dynamic):
            self.checkForCongestion(dla=dla, maxCongestion=maxCongestion)
        action = []
        for e in [e for e in self.elems if isinstance(e, Diverter)]:
            eID_src = [e.ID for e in self.elems if isinstance(e, Diverter)].index(e.ID)
            if e.tote is not None:
                eID_dst = [e_.ID for e_ in self.elems if e_.ID in self.src+self.dst].index(e.tote.dst)
            else:
                eID_dst = 0
            if e.forced_control is None:
                action.append(int(self.shortestPathTable[eID_src][eID_dst]))
            else:
                action.append(int(e.forced_control))
        return(action)
        
    def reset(self, total=True, seed=None, numtotes=None):
        if numtotes is not None:
            self.numtotes=numtotes
        if seed is None:
            seed = self.default_rand.choice(range(1000000))
        self.rand_dst = random.Random(seed)
        self.rand_src = random.Random(seed+1)
        self.rand_numtotes = random.Random(seed+2)
        if self.randomize_numtotes:
            self.numtotes = self.rand_numtotes.randint(1,len(self.elems))

        obs = self.getObs()

        self.stepnumber = 0
        if total:
            obs = []
            self.tote_info = {}
            for e in self.elems:
                e.tote=None
                obs.append([0])
                
                if isinstance(e, Toploader):
                     e.totes = []
                     e.tote = None
                   
            self.totes = []
            self.addTotes(self.numtotes)
            self.shortestPathTable = self.calcShortestPathTable()
        else:
            for key in self.tote_info:
                self.tote_info[key]['TotalSteps'] = 0
                self.tote_info[key]['Destinations'] = [t.dst for t in self.totes if t.ID == key]
                self.tote_info[key]['StepsPerDst'] = [0]
                self.tote_info[key]['DstReached'] = [0]
                self.tote_info[key]['StepNumber'] = [self.stepnumber]
        
        self.congestion=False
        self.done = False
        return np.array(obs)
    
    def step(self,action=[], shortestPath=False, dynamic=False, dla=False, maxCongestion=1.0): 
        # self.translist = []
#        start = time.time()
#        "the code you want to test stays here"
        
#        for e in self.elems:
#            if e.tote is not None:
#                print(e.ID, e.tote.ID, e.tote.dst)
#            else:
#                print(e.ID, ' ', ' ')
#        
        self.deadlock=False
        if shortestPath:
            action = self.calcShortestPath(dynamic=dynamic, dla=dla, maxCongestion=maxCongestion)
        elif dla:
            action = self.performDLA(action=action, maxCongestion=maxCongestion)
        reward = 0
        deadlock = False
        diverters = [e for e in self.elems if isinstance(e, Diverter)]

        e_ready = [e_ for e_ in self.elems if e_.tote is not None and not e_.tote.moved]
        e_old_1 = e_ready.copy()
        while e_ready != []:
            e_old_2 = e_ready.copy()
            for e in e_ready:
                if e in diverters:
                    e.move(control=action[diverters.index(e)])
                else:
                    e.move(control=0)
                if e.tote is None or e.tote.moved:
                    e_ready.remove(e)
            if e_ready == e_old_2:
#                self.congestion_counter += 1
#                print("Local gridlock")
                break
#            else:
#                self.congestion_counter = 0
        if e_ready == e_old_1:# and self.congestion_counter:
            print('Deadlock detected!')
            deadlock = True
            
        for t in self.totes:
            if t == t.element.tote: # ensures that tote is not in queue to toploader "outside" environment
                self.tote_info[t.ID]['TotalSteps']+=1
                self.tote_info[t.ID]['StepsPerDst'][-1]+=1
                self.tote_info[t.ID]['StepNumber'][-1]=self.stepnumber

            t.moved=False
            
            if t.dst == t.element.ID and t.dst in self.dst:
                reward += 1
                self.tote_info[t.ID]['DstReached'][-1]=1
                self.tote_info[t.ID]['Destinations'][-1] = t.element.ID # to store the used source
                self.setDestination(t)
            elif t.element.ID in self.src and t.dst in self.src:
                self.tote_info[t.ID]['DstReached'][-1]=1
                self.setDestination(t)
        obs = self.getObs()
        tote_info = self.tote_info.copy()
        if deadlock:
            _ = self.reset(total=True)
            self.deadlock=True
            self.done = True
        self.stepnumber += 1
        if (self.stepnumber >= self.steplimit):
            self.done = True
        return obs, reward, self.done, tote_info

