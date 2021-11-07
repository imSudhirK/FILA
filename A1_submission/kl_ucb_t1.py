import numpy as np


class KL_UCB():
    def __init__(self, Pa0, ep, u_at, emp_Pa):
        self.Pa0 = Pa0                         #true probability of success for arm a
        self.ep = ep
        self.u_at = u_at                     # number of pulls for arm a till t steps
        self.emp_Pa  = emp_Pa                  # emperical mean of arm a till at t
        self.cumul_reward = 0
        self.horizon_T = 0

#-------------------------------------------initialize number of pulls and empirical mean
    def initializeKL_UCB(self, n_arms):
        self.u_at = [0 for _ in range(n_arms)]       #set, number of pulls for each arm = 0
        self.emp_Pa = [0 for _ in range(n_arms)]      #set, emperical mean for each arm = 0

#-------------------------------------KL - definition function
    def KL(self, x ,y):
        if x==y:
            return 0
        if x==0:
            return np.log(1/(1-y))
        return x*np.log(x/y)+(1-x)*np.log((1-x)/(1-y))

#----------------------------find maximum of q from given range which satisfy conditions
    def ret_maxQ(self, emp_Pa, t_step, u_at_a, c, pre):
        q_min = emp_Pa
        q_max = 1
        q_ret = (float(emp_Pa)+1.0) / 2.0
        log_term = np.log(t_step)+ c*np.log(np.log(t_step))
        while(True):
            uatXlogterm = u_at_a*self.KL(emp_Pa, q_ret)
            if(uatXlogterm <= log_term):
                if(log_term < pre + uatXlogterm):
                    return q_ret
                q_min= q_ret
                q_ret = (q_ret + q_max)/2.0
            else:
                q_max  = q_ret
                q_ret = (q_ret + q_min)/2.0

#--------------------------------------pull arm greedy with highest KL_UCB
    def pull_A(self, pre = 0.001, c=3):
        n_A = len(self.u_at)                         #total number of arms
        for a in range(n_A):                         #pull each arm atleast once
            if self.u_at[a] == 0:
                return a
        t_step= sum(self.u_at)+1                      # current time step
        kl_ucb_a = [float(0) for _ in range(n_A)]      #initialize ucb for each arm = 0
        for a in range(n_A):                         # calculate ucb at t-th step
            u_at_a = float(self.u_at[a])
            emp_Pa = float(self.emp_Pa[a])
            if emp_Pa == 1.0:
                kl_ucb_a[a] = emp_Pa
            else:
                q_max = self.ret_maxQ(emp_Pa, t_step, u_at_a, c, pre)
                kl_ucb_a[a] = q_max
        return np.argmax(kl_ucb_a)

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
