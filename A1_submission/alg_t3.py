import numpy as np

class ThompSamp_t3():
    def __init__(self, Pa0, ep, u_at, s_at, f_at, emp_Pa):
        self.Pa0 = Pa0                         #true probability of success for arm a
        self.ep = ep
        self.u_at = u_at                     # number of pulls for arm a till t steps
        self.s_at = s_at
        self.f_at = f_at
        self.emp_Pa  = emp_Pa                  # emperical mean of arm a till at t
        self.cumul_reward = 0
        self.horizon_T = 0

#-------------------------------------------initialize values
    def initializeThmpS(self, n_arms):
        self.u_at = [0 for _ in range(n_arms)]       #set, number of pulls for each arm = 0
        self.s_at = [0 for _ in range(n_arms)]
        self.f_at = [0 for _ in range(n_arms)]
        self.emp_Pa = [0.0 for _ in range(n_arms)]      #set, emperical mean for each arm = 0


#-------------------------------------------find ThmpS greedy arm to pull
    def pull_A(self, prec = 0.001, c = 3):
        n_A = len(self.u_at)                         #total number of arms
        beta_rand = [float(0) for _ in range(n_A)]
        for a in range(n_A):                         #pull each arm atleast once
            s_at_a = float(self.s_at[a])
            f_at_a = float(self.f_at[a])
            beta_rand[a] = np.random.beta(s_at_a+1, f_at_a+1)
        return np.argmax(beta_rand)


#--------------------------------update empirical mean for arm a at t-th if pulled
    def updt_emp_Pa(self, a, r):                    # a - greedy arm , r - reward of that arm
        self.u_at[a] +=1
        if r ==1:
            self.s_at[a] += 1
        else:
            self.f_at[a] += 1
        self.emp_Pa[a] = ((self.u_at[a]-1)*float(self.emp_Pa[a]) + r)/float(self.u_at[a])

#------------------------------reward after pulling arm a for Bernoulli bandit instances
    def ret_reward(self, a):
        p = np.random.uniform(0, 1)
        return p
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
