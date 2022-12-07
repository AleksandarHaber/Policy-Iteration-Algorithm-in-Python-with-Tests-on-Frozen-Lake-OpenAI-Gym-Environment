# -*- coding: utf-8 -*-
"""
Functions for implementing the policy iteration algorithm

Author: Aleksandar Haber
Date: December 2022

"""
##################
# this function computes the state value function by using the iterative policy evaluation algorithm
##################
# inputs: 
##################
# env - environment 
# valueFunctionVector - initial state value function vector
# policy - policy to be evaluated - this is a matrix with the dimensions (number of states)x(number of actions)
#        - p,q entry of this matrix is the probability of selection action q in state p
# discountRate - discount rate 
# maxNumberOfIterations - max number of iterations of the iterative policy evaluation algorithm
# convergenceTolerance - convergence tolerance of the iterative policy evaluation algorithm
##################
# outputs:
##################
# valueFunctionVector - final value of the state value function vector 

##################
def evaluatePolicy(env,valueFunctionVector,policy,discountRate,maxNumberOfIterations,convergenceTolerance):
    import numpy as np
    convergenceTrack=[]
    for iterations in range(maxNumberOfIterations):
        convergenceTrack.append(np.linalg.norm(valueFunctionVector,2))
        valueFunctionVectorNextIteration=np.zeros(env.observation_space.n)
        for state in env.P:
            outerSum=0
            for action in env.P[state]:
                innerSum=0
                for probability, nextState, reward, isTerminalState in env.P[state][action]:
                    #print(probability, nextState, reward, isTerminalState)
                    innerSum=innerSum+ probability*(reward+discountRate*valueFunctionVector[nextState])
                outerSum=outerSum+policy[state,action]*innerSum
            valueFunctionVectorNextIteration[state]=outerSum
        if(np.max(np.abs(valueFunctionVectorNextIteration-valueFunctionVector))<convergenceTolerance):
            valueFunctionVector=valueFunctionVectorNextIteration
            print('Iterative policy evaluation algorithm converged!')
            break
        valueFunctionVector=valueFunctionVectorNextIteration       
    return valueFunctionVector

##################
# this function computes an improved policy 
##################
# inputs: 
# env - environment 
# valueFunctionVector - state value function vector that is previously computed
# numberActions - number of actions 
# numberStates - number of states 
# discountRate - discount rate

# outputs:
# improvedPolicy - improved policy
# qvaluesMatrix  - matrix containing computed action-value functions 
#                - (p,q) entry of this matrix is the action value function computed at the state p and for the action q
# Note: qvaluesMatrix is just used for double check - it is actually not used lated on    
    
    
##################

def improvePolicy(env,valueFunctionVector,numberActions,numberStates,discountRate):
    import numpy as np
    # this matrix will store the q-values (action value functions) for every state
    # this matrix is returned by the function 
    qvaluesMatrix=np.zeros((numberStates,numberActions))
    # this is the improved policy
    # this matrix is returned by the function
    improvedPolicy=np.zeros((numberStates,numberActions))
    
    for stateIndex in range(numberStates):
        # computes a row of the qvaluesMatrix[stateIndex,:] for fixed stateIndex, 
        # this loop iterates over the actions
        for actionIndex in range(numberActions):
            # computes the Bellman equation for the action value function
            for probability, nextState, reward, isTerminalState in env.P[stateIndex][actionIndex]:
                qvaluesMatrix[stateIndex,actionIndex]=qvaluesMatrix[stateIndex,actionIndex]+probability*(reward+discountRate*valueFunctionVector[nextState])
            
        # find the action indices that produce the highest values of action value functions
        bestActionIndex=np.where(qvaluesMatrix[stateIndex,:]==np.max(qvaluesMatrix[stateIndex,:]))

        # form the improved policy        
        improvedPolicy[stateIndex,bestActionIndex]=1/np.size(bestActionIndex)
    return improvedPolicy,qvaluesMatrix
