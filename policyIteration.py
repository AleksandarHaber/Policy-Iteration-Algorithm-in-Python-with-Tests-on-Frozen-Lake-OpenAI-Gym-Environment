# -*- coding: utf-8 -*-
"""
Reinforcement Learning Tutorial:
    
Implementation of the Policy Iteration Algorithm in Python 

Tested on the Frozen Lake OpenAI Gym environment.

Author: Aleksandar Haber 
Date: December 2022 

"""
import gym
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from functions import evaluatePolicy
from functions import improvePolicy

# create the environment 
# this is a completely deterministic environment
env=gym.make('FrozenLake-v1', desc=None, map_name="4x4", is_slippery=False,render_mode="human")
# this is a completely stochastic environment - the algorithm will not work properly since the transition probabilities are equal -too much!
#env=gym.make("FrozenLake-v1", render_mode="human") 
env.reset()
# render the environment
# uncomment this if you want to render the environment
env.render()
#
env.close()

# investigate the environment
# observation space - states 
env.observation_space

env.action_space
# actions:
#0: LEFT
#1: DOWN
#2: RIGHT
#3: UP

##########################################################################
#           general parameters for the policy iteration
##########################################################################
# select the discount rate
discountRate=0.9
# number of states - determined by the Frozen Lake environment
stateNumber=16
# number of possible actions in every state - determined by the Frozen Lake environment
actionNumber=4
# maximal number of iterations of the policy iteration algorithm 
maxNumberOfIterationsOfPolicyIteration=1000

# select an initial policy
# initial policy starts with a completely random policy
# that is, in every state, there is an equal probability of choosing a particular action
initialPolicy=(1/actionNumber)*np.ones((stateNumber,actionNumber))
##########################################################################
#           parameters of the iterative policy evaluation algorithm
##########################################################################
# initialize the value function vector
valueFunctionVectorInitial=np.zeros(env.observation_space.n)
# maximum number of iterations of the iterative policy evaluation algorithm
maxNumberOfIterationsOfIterativePolicyEvaluation=1000
# convergence tolerance 
convergenceToleranceIterativePolicyEvaluation=10**(-6)
###########################################################################
###########################################################################

for iteration in range(maxNumberOfIterationsOfPolicyIteration):
    print("Iteration - {} - of policy iteration algorithm".format(iteration))
    if (iteration == 0):
        currentPolicy=initialPolicy
    valueFunctionVectorComputed =evaluatePolicy(env,valueFunctionVectorInitial,currentPolicy,discountRate,maxNumberOfIterationsOfIterativePolicyEvaluation,convergenceToleranceIterativePolicyEvaluation)
    improvedPolicy,qvaluesMatrix=improvePolicy(env,valueFunctionVectorComputed,actionNumber,stateNumber,discountRate)
    # if two policies are equal up to a certain "small" tolerance
    # then break the loop - our algorithm converged
    if np.allclose(currentPolicy,improvedPolicy):
        currentPolicy=improvedPolicy
        print("Policy iteration algorithm converged!")
        break
    currentPolicy=improvedPolicy
    
    


# test to determine the indices of max entries of an array
#aa1=np.array([0,2,2,1])
#test1=np.where(aa1==np.max(aa1))
        
