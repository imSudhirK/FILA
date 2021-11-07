import numpy as np

class valueIteration():
    def __init__(self, numS, numA, transitions, gamma):
        self.numS = numS                  #number of sates
        self.numA = numA                  #number of actions
        self.gamma = gamma           #discount factor
        self.transitions = transitions
        self.v = np.zeros(numS)            #value
        self.optimalA = np.zeros(numS)     #policy

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

#----------------------------value function V_pi(s) for value interation
    def VF(self, s):
        avf = self.Q(s, self.v)
        self.optimalA[s] = np.argmax(avf)
        return np.max(avf)

#---------------------------- run untill pricision
    def Run_untill(self, precision = 0.00000001):
        diff_v = np.zeros(self.numS)
        while(1):
            for s in range(self.numS):    #update value
                v_curr = self.VF(s)
                diff_v[s] = np.abs(v_curr - self.v[s])
                self.v[s] = v_curr
            max_e = 0
            for s in range(self.numS):    #max error
                max_e = max(max_e, diff_v[s])
            if max_e < precision:
                break

#-----------------------printing optimal value function and optimal policy
    def VI_print(self):
        for s in range(self.numS):
            print("{}\t{}".format(self.v[s], int(self.optimalA[s])))
