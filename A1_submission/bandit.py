import numpy as np
import matplotlib.pyplot as plt
import os, sys, argparse, time

#--------------------------import algorithm class from files
from ucb_t1 import UCB
from kl_ucb_t1 import KL_UCB
from epsilon_greedy_t1 import eG3
from thompson_sampling_t1 import ThompSamp

from ucb_t2 import UCB_t2

#----------------------------main function
if __name__ == "__main__":
    t_step = time.time()

    arg_parse = argparse.ArgumentParser()
    arg_parse.add_argument("--instance", help = "enter path to the instance file")
    arg_parse.add_argument("--algorithm", help = "enter one of given algorithm")
    arg_parse.add_argument("--randomSeed", help = "enter randomSeed: non-negative integer")
    arg_parse.add_argument("--epsilon", help = "enter ep: a number in [0, 1]")
    arg_parse.add_argument("--scale", help = "enter c: a positive real number")
    arg_parse.add_argument("--threshold", help = "enter th: a number in [0, 1].")
    arg_parse.add_argument("--horizon", help = "enter hz: a non-negative integer")

    HIGHS = 0
    epsilon = 0.02
    scale = 2
    threshold = 0
    arguments = arg_parse.parse_args()

#-----------------------------------type cast default string arguments
    path_instance = arguments.instance
    algos = arguments.algorithm
    randSeed = int(arguments.randomSeed)
    if arguments.epsilon:
        epsilon = float(arguments.epsilon)
    if arguments.scale:
        scale = float(arguments.scale)
    if arguments.threshold:
        threshold = float(arguments.threshold)
    horizon = int(arguments.horizon)

#---------------------------- read files and set true means
    handler = open(path_instance)
    l= handler.readlines()
    bandits_instance =[float(val) for val in l]
    handler.close()

    np.random.seed(randSeed)

#----------------------------- running algorithms as per input
    if(algos == "epsilon-greedy-t1"):
        ep_greedy = eG3(bandits_instance, epsilon, [], [])
        ep_greedy.initialize_eG3(len(bandits_instance))
        ep_greedy.resume_untill(horizon)
        REG = ep_greedy.ret_regret()
        print("{}, {}, {}, {}, {}, {}, {}, {}, {}".format(path_instance, algos,\
         randSeed, epsilon, scale, threshold, horizon, REG, HIGHS))

    if(algos == "ucb-t1"):
        ucb_greedy = UCB(bandits_instance, epsilon, [], [])
        ucb_greedy.initializeUCB(len(bandits_instance))
        ucb_greedy.resume_untill(horizon)
        REG = ucb_greedy.ret_regret()
        print("{}, {}, {}, {}, {}, {}, {}, {}, {}".format(path_instance, algos,\
         randSeed, epsilon, scale, threshold, horizon, REG, HIGHS))

    if(algos == "kl-ucb-t1"):
        klucb_greedy = KL_UCB(bandits_instance, epsilon, [], [])
        klucb_greedy.initializeKL_UCB(len(bandits_instance))
        klucb_greedy.resume_untill(horizon)
        REG = klucb_greedy.ret_regret()
        print("{}, {}, {}, {}, {}, {}, {}, {}, {}".format(path_instance, algos,\
         randSeed, epsilon, scale, threshold, horizon, REG, HIGHS))

    if(algos == "thompson-sampling-t1"):
        tmpS_greedy = ThompSamp(bandits_instance, epsilon, [], [], [], [])
        tmpS_greedy.initializeThmpS(len(bandits_instance ))
        tmpS_greedy.resume_untill(horizon)
        REG = tmpS_greedy.ret_regret()
        print("{}, {}, {}, {}, {}, {}, {}, {}, {}".format(path_instance, algos,\
         randSeed, epsilon, scale, threshold, horizon, REG, HIGHS))

#------------------------------------------------for task2
    if(algos == "ucb-t2"):
        ucb_greedy = UCB_t2(bandits_instance, epsilon, [], [], scale)
        ucb_greedy.initializeUCB(len(bandits_instance))
        ucb_greedy.resume_untill(horizon)
        REG = ucb_greedy.ret_regret()
        print("{}, {}, {}, {}, {}, {}, {}, {}, {}".format(path_instance, algos,\
         randSeed, epsilon, scale, threshold, horizon, REG, HIGHS))
