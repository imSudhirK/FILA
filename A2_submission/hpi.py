import numpy as np

class howardsPolicyIteration():
    def __init__(self, numS, numA, transitions, gamma):
        self.numS = numS                   #number of states
        self.numA = numA                   #number of actions
        self.gamma = gamma                 #discount factor
        self.transitions = transitions
        self.v = np.zeros(numS)            #value
        self.optimalA = []                 #policy
        #------------------------------initialize optimalA
        for s in range(self.numS):
            t=[]
            for a in range(len(self.transitions[s])):
                if len(self.transitions[s][a])!=0:
                    t.append(a)
            if (len(t))!=0:
                self.optimalA.append(t[0])
            else:
                self.optimalA.append(0)

#-----------------------------action value function
    def Q(self, s, v):
        avf = np.zeros(self.numA)
        for a in range(self.numA):
            for T in self.transitions[s][a]:      #summation
                s_ = T[0]
                Tsas_ = T[2]
                Rsas_ = T[1]
                avf[a]= avf[a] + (Tsas_*(Rsas_ + self.gamma*v[s_]))
        return avf

#--------------------------------optimal value for optimal policy
    def VS(self, pi):
        A_mat = []                #A_mat* X_mat = B_mat
        B_mat = [0]*self.numS
        for _ in range(self.numS):
            A_mat.append([0]*self.numS)
        for s in range(self.numS):
            a = pi[s]                #action
            A_mat[s][s] = 1
            c = 0
            for t in self.transitions[s][a]:
                s_ = t[0]
                Tsas_ = t[2]
                Rsas_ = t[1]
                s_coeff = self.gamma*Tsas_
                A_mat[s][s_] = A_mat[s][s_]-1*s_coeff
                c = c + Tsas_*Rsas_
            B_mat[s] = c
        X_mat = np.linalg.solve(A_mat, B_mat)
        return X_mat

#---------------------------------run untill every states visited
    def Run_untill(self):
        optA = self.optimalA
        while(1):
            V_pi = self.VS(optA)
            flag = 0
            for s in range(self.numS):
                avf = self.Q(s, V_pi)
                a = self.optimalA[s]    #action
                q_val = avf[a]
                for i_a in range(self.numA):
                    if avf[i_a] > q_val:
                        a = i_a
                        break
                if(a == optA[s]):
                    flag = flag + 1
                else:
                    optA[s] = a
                    self.optimalA[s] = a
            if flag == self.numS:
                self.v = V_pi
                break

#-----------------------printing optimal value function and optimal policy
    def HPI_print(self):
        for s in range(self.numS):
            print("{}\t{}".format(self.v[s], int(self.optimalA[s])))
