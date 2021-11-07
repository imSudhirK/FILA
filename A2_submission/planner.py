import numpy as np
import time, argparse
from vi import valueIteration
from hpi import howardsPolicyIteration
from lp import linearProgramming

if __name__=="__main__":
    t = time.time()

    parser = argparse.ArgumentParser()
    parser.add_argument("--mdp", help = "mdp file")
    parser.add_argument("--algorithm", help = "algorithm lp, vi or hpi")
    arguments = parser.parse_args()

#---------------------------typecast all default string arguments
    mdp_file = arguments.mdp
    algos = arguments.algorithm
    if algos == None:            #default
        algos = "lp"
    np.random.seed(0)

#---------------------------read files/parsing and initialize fields
    fileHandler = open(mdp_file)
    l = fileHandler.readlines()
    nl = len(l)
    states = int(l[0].rstrip().split(" ")[1])
    actions = int(l[1].rstrip().split(" ")[1])
    discount = float(l[nl-1].rstrip().split(" ")[-1])
    mdptype = l[nl-2].rstrip().split(" ")[-1]
    transitions = []
    for i in range(states):
        Trans = []
        for j in range(actions):
            Trans.append([])
        transitions.append(Trans)
    for i in range(4,nl-2):
        Trans = l[i].rstrip().split(" ")
        transitions[int(Trans[1])][int(Trans[2])].append([int(Trans[3]),float(Trans[4]),float(Trans[5])])
    fileHandler.close()

#-----------------------------running algorithms as per input
    if(algos == "vi"):
        VI = valueIteration(states, actions, transitions, discount)
        VI.Run_untill()
        VI.VI_print()

    if(algos == "lp"):
        Lp = linearProgramming(states, actions, transitions, discount)
        Lp.Run_untill()
        Lp.LP_print()
        pass

    if(algos == "hpi"):
        HPI = howardsPolicyIteration(states, actions, transitions, discount)
        HPI.Run_untill()
        HPI.HPI_print()
