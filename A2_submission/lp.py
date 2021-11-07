import numpy as np
from pulp import *

class linearProgramming():
    def __init__(self, numS, numA, transitions, gamma):
        self.numA = numA                       #number of actions
        self.numS = numS                       #number of states
        self.gamma = gamma                     #discount factor
        self.transitions = transitions
        self.v= np.zeros(numS)                 #value
        self.optimalA = np.zeros(self.numS)    #policy

#---------------------expected long term reward starting at s/action value function
    def Q(self, s, v):
        avf = np.zeros(self.numA)
        for a in range(self.numA):
            for T in self.transitions[s][a]:     #summation
                s_ = T[0]
                Tsas_ = T[2]
                Rsas_ = T[1]
                avf[a]= avf[a] + (Tsas_*(Rsas_ + self.gamma*v[s_]))
        return avf

#----------------------------value function for linear Programming
    def VF(self):
        v_ = LpVariable.dicts("v", range(self.numS))
        Obj = LpProblem("lp", LpMinimize)                    #objective
        Obj += lpSum([v_[i] for i in range(self.numS)])
        for s in range(self.numS):                  #adding nk constaints and solving
            for a in range(self.numA):
                vf = 0                               #value func
                for t in self.transitions[s][a]:
                    s_ = t[0]
                    Tsas_ = t[2]
                    Rsas_ = t[1]
                    vf = vf + (Tsas_*(Rsas_ + self.gamma*v_[s_]))
                Obj += v_[s] >= vf                   # adding this states paticular action constraint

        LpSolverDefault.msg = 0
        Obj.solve()
        V_pi = np.zeros(self.numS)
        for v in Obj.variables():
            k = v.name.split("_")[1]
            V_pi[int(k)] = v.varValue
        return V_pi

#----------------------------run/ update vals for each states
    def Run_untill(self):
        v_pi = self.VF()
        for s in range(self.numS):
            avf = self.Q(s, v_pi)
            optA = np.argmax(avf )
            self.optimalA[s] = optA
            self.v[s] = v_pi[s]

#-----------------------printing optimal value function and optimal policy
    def LP_print(self):
        for s in range(self.numS):
            print("{}\t{}".format(self.v[s], int(self.optimalA[s])))
