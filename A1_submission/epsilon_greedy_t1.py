import numpy as np

class eG3():
    def __init__(self, Pa0, ep, u_at, emp_Pa):
        self.Pa0 = Pa0                         #true probability of success for arm a
        self.ep = ep
        self.u_at = u_at                     # number of pulls for arm a till t steps
        self.emp_Pa  = emp_Pa                  # emperical mean of arm a till at t
        self.cumul_reward = 0
        self.horizon_T = 0

#-------------------------------------------initialize number of pulls and empirical mean
    def initialize_eG3(self, n_arms):
        self.u_at = [0 for _ in range(n_arms)]       #set, number of pulls for each arm = 0
        self.emp_Pa = [0 for _ in range(n_arms)]      #set, emperical mean for each arm = 0

#-------------------------------------------find arm with highest ucb
    def pull_A(self):
        n_A = len(self.u_at)                         #total number of arms
        for a in range(n_A):                         #pull each arm atleast once
            if self.u_at[a] == 0:
                return a
        prob = np.random.random()
        if prob < self.ep:
            return np.random.randint(len(self.emp_Pa))
        else:
            return np.argmax(self.emp_Pa)

#--------------------------------update empirical mean for arm a at t-th if pulled
    def updt_emp_Pa(self, a, r):                    # a - greedy arm , r - reward of that arm
        self.u_at[a] +=1                            # update number of pulls for arm a
        self.emp_Pa[a] = ((self.u_at[a]-1)*float(self.emp_Pa[a]) + r)/float(self.u_at[a])

#------------------------------reward after pulling arm a for Bernoulli bandit instances
    def ret_reward(self, a):
        p = np.random.random()
        if self.Pa0[a] < p:
            return 0.0
        else:
            return 1.0

#-----------------------------continue till given horizon T
    def resume_untill(self, T):
        self.horizon_T = T
        for _ in range(T):
            a = self.pull_A()
            r = self.ret_reward(a)
            self.cumul_reward += r
            self.updt_emp_Pa(a, r)

#-------------------------------regret of algorithm
    def ret_regret(self):
        P_star = max(self.Pa0)
        regret = P_star*self.horizon_T - self.cumul_reward
        return regret
