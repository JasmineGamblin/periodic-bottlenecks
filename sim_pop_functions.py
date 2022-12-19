#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 10 14:58:01 2022

@author: jasminegamblin
"""

import numpy as np
import matplotlib.pyplot as plt




def poisson (l) :
    """
    Draw a Poisson random variable for large l
    Arguments:
        l: parameter of Poisson distribution
    Returns:
        a random number drawn from a Gaussian distribution with parameters
        (l,sqrt(l))
    """
    return np.random.normal(l, np.sqrt(l))




def binomial (n, p) :
    """
    Draw a binomial random variable for large n and small p
    Arguments:
        n, p: parameters of binomial distribution
    Returns:
        a random number drawn from a Poisson distribution with parameter n*p,
        or a Gaussian distribution with parameters (n*p, sqrt(n*p)) if n*p is
        large
    """
    if n*p > 10**18 :
        return poisson(n*p)
    else :
        return np.random.poisson(n*p)




def gillespie (v_old, t_old, rates, mu, tau, K) :
    """
    Updates the Gillespie process: selects the time of the next event (birth,
    death or mutation), then selects the type of event and updates
    subpopulation sizes accordingly
    Arguments:
        v_old: array of subpopulation sizes at time t_old
        t_old: time
        rates: list of intrinsic birth and death rates of the 5 subpopulations
        mu: list of the 2 mutation rates
        tau: tau-leaping parameter
        K: list of carrying capacities
    Returns:
        v_new: updated array of subpopulation sizes
        t_new: incremented time
    """
    mut = [sum(mu), mu[1], mu[0], 0, 0] # list of mutation rates for each of the 5 subpopulations
    
    # rates
    if K : # if the growth is density-dependent, multiply intrinsic birth and death rates by 1-pop_size/K
        pop_size = sum(v_old)
        rate_log = np.copy(rates)
        rate_log[0] = [rates[0][i]*max(0,1-pop_size/K[i]) for i in range(5)]
        rate_log[1] = [rates[1][i]*max(0,1-pop_size/K[i]) for i in range(5)]
        birth_rates2 = [v_old[i]*rate_log[0][i]*(1-mut[i]) for i in range(5)]
        mut_rates2 = [v_old[i]*rate_log[0][i]*mut[i] for i in range(5)]
        death_rates2 = [v_old[i]*rate_log[1][i] for i in range(5)]
    else :
        birth_rates2 = [v_old[i]*rates[0][i]*(1-mut[i]) for i in range(5)]
        mut_rates2 = [v_old[i]*rates[0][i]*mut[i] for i in range(5)]
        death_rates2 = [v_old[i]*rates[1][i] for i in range(5)]
    
    # cumulative rates
    birth_rates = np.cumsum(birth_rates2)                               
    mut_rates = np.add(np.cumsum(mut_rates2), birth_rates[-1])
    death_rates = np.add(np.cumsum(death_rates2), mut_rates[-1])
    
    # case where all subpopulation sizes are 0
    if death_rates[-1] == 0 : 
        return v_old, np.inf
    
    # case where rates are not too high
    elif death_rates[-1] < 100 : 
        # time increment
        dt = np.random.exponential(1/death_rates[-1])
        t_new = t_old + dt
        v_new = np.copy(v_old)
        
        # next event
        u = np.random.uniform(low = 0, high = death_rates[-1])
        if (u < birth_rates[-1]) : # next event is a birth
            j = min(np.where(birth_rates >= u)[0])
            v_new[j] += 1
        elif (u < mut_rates[-1]) : # next event is a mutation
            j = min(np.where(mut_rates >= u)[0]) # mutation happenned in subpopulation j
            if j==0 :
                k = np.random.choice(np.arange(1,3), p = [mu[0]/sum(mu), mu[1]/sum(mu)])
                v_new[k] += 1
            elif j==1 :
                v_new[3] += 1
            elif j==2 :
                v_new[4] += 1
        else : # next event is a death
            j = min(np.where(death_rates >= u)[0])
            v_new[j] -= 1
        return v_new, t_new 
    
    # case where rates are high -> tau-leaping
    else : 
        dt = tau
        v_new = np.copy(v_old)
        
        # next events
        neg = True
        while neg :
            # birth and death events
            incr = [0]*5
            for i in range(5) :
                if dt*(birth_rates2[i] + death_rates2[i]) > 10**18 :
                    # use normal approximation to prevent overflow
                    incr[i] = poisson(dt*birth_rates2[i]) - poisson(dt*death_rates2[i])
                else :
                    incr[i] = np.random.poisson(dt*birth_rates2[i]) - np.random.poisson(dt*death_rates2[i])
            v_new = np.add(v_old, incr)
            if (np.array(v_new) < 0).any() : # check that no population size is < 0
                dt = dt/2
            else :
                # mutation events
                nb_mut = [0]*3
                for i in range(3) :
                    if dt*mut_rates2[i] > 10**18 :
                        # use normal approximation to prevent overflow
                        nb_mut[i] = poisson(dt*mut_rates2[i])
                    else :
                        nb_mut[i] = np.random.poisson(dt*mut_rates2[i])
                if nb_mut[0] > 10**18 :
                    # use poisson or normal approximation to prevent overflow
                    mut1 = binomial(nb_mut[0], mu[0]/sum(mu))
                else :
                    mut1 = np.random.binomial(nb_mut[0], mu[0]/sum(mu))
                v_new[1] += mut1
                v_new[2] += nb_mut[0] - mut1   
                v_new[3] += nb_mut[1]
                v_new[4] += nb_mut[2]
                neg = False
                
        # time increment
        t_new = t_old + dt
        return v_new, t_new




def sim_cycle (tn, start, rates, mu, tau, K = None) :
    """
    Simulates population evolution during one growth phase
    Arguments:
        tn: time duration of the growth phase
        start: subpopulation sizes at the beginning of growth phase
        rates: list of intrinsic birth and death rates of the 5 subpopulations
        mu: list of the 2 mutation rates
        tau: tau-leaping parameter
        K: list of carrying capacities
    Returns:
        t: list of event times (t[i] is the time at step i)
        v: 2-dimensional array of subpopulation sizes at
        these times (v[i][j] is the pop size of
        subpopulation j at step i)
    """
    # initialisation
    v = [np.array(start, dtype = float)]
    t = [0]
    
    # update using Gillespie's algorithm
    while t[-1] < tn :
        v_new, t_new = gillespie(v[-1], t[-1], rates, mu, tau, K)
        v.append(v_new)
        t.append(t_new)
        
    if t[-1] != np.inf :
        v[-1] = np.copy(v[-2])
    t[-1] = tn
    return t, v




def scenario (paths) :
    """
    Attributes a scenario number according to observed evolutionary paths
    Argument:
        paths: observed evolutionary paths during a simulation, encoded by a
        list of 4 binary numbers
    Returns:
        scenario: the number corresponding to the observed scenario
        (which will be used for plotting information or color)
    """
    scenario = -1 # no mutant detected
    if paths == [1,0,0,0] : # mutant 10 detected
        scenario = 0
    elif paths == [1,0,1,0] : # mutants 10 and 11 (from 10) detected
        scenario = 1
    elif paths == [1,1,0,0] : # mutants 10 and 01 detected
        scenario = 2
    elif paths == [1,1,1,0] : # mutants 10, 01 and 11 (from 10) detected
        scenario = 3
    elif paths == [1,1,0,1] : # mutants 10, 01 and 11 (from 01) detected
        scenario = 4
    elif paths == [1,1,1,1] : # mutants 10, 01 and 11 (from both 10 and 01) detected
        scenario = 5
    return scenario




def sim_pop (n, nb_cycles, alpha, beta, delta, rates, tau = 0.05, p = None, s = 0.1, scenario_only = False) :
    """
    Simulates the evolution of a bottlenecked population for several cycles
    Arguments:
        n: n parameter (inverse of total mutation rate)
        nb_cycles: number of cycles (growth + bottleneck) to simulate
        alpha: parameter for final wild-type population size
        beta: parameter for initial wild-type population size
        delta: parameter for low mutation rate
        rates: list of intrinsic birth and death rates of the 5 subpopulations
        tau: tau-leaping parameter
        p: proportion of carrying capacity reached by the WT before dilution
        (set to None to make a simulation without density-dependence)
        s: mutants' advantage for resource exploitation (not used if p=None)
        scenario_only: set to True to stop the simulation when the final
        scenario has been reached
    Returns:
        t: list of event times (t[i] is the time at step i)
        v: 2-dimensional array of subpopulation sizes at these times (v[i][j]
        is the pop size of subpopulation j at step i)
        scenario: number of the scneario observed during this simulation
        cycle: cycle where double mutants where observed for the first time
        (-1 if they never appear)
    """
    mu = [1/n, 1/(n**delta)]
    r00 = rates[0][0]*(1 - sum(mu)) - rates[1][0]    
    if p :
        tn = np.log((n**(alpha-beta) - 1)/(1/p-1))/r00 # duration of the growth phase
        Dn = n**(beta-alpha)/p # dilution factor
        K = [n**alpha] + 4*[n**(alpha*(1+s))] # carrying capacities
    else :
        tn = np.log(n)*(alpha-beta)/r00 
        Dn = n**(beta-alpha)
    threshold = 50 # threshold for mutant emergence
    
    # initialisation
    start = [0]*5
    start[0] = n**beta
    v = np.array(start, ndmin = 2)
    t = [0]
    c = 0 # current cycle
    paths = [0]*4 # observed evolutionary paths
    cycle = -1 # cycle of double mutant emergence
    
    # simulate nb_cycles
    while c < nb_cycles and sum(start)>0 and ((not scenario_only) or (sum(start[0:2]) > 0 and sum(paths) < 4)) :
        # simulation of growthg phase
        if p :
            sim = sim_cycle(tn, start, rates, mu, tau, K = K)
        else :
            sim = sim_cycle(tn, start, rates, mu, tau)
        
        # bottleneck event
        start = [0]*5
        for i,x in enumerate(sim[1][-1]) :
            if (x > 10**18) :
                # approximate by poisson distribution to prevent overflow
                start[i] = binomial(x, Dn)
            else :
                start[i] = np.random.binomial(x, Dn)
                
        # update v, t and c
        v = np.concatenate((v, np.array(sim[1][1:]), np.array(start, ndmin = 2)), axis = 0)
        t += [t[-1] + x for x in sim[0][1:]] + [t[-1]+sim[0][-1]]
        c += 1
        
        # check which mutants are present above threshold
        end_cycle =  sim[1][-1][1:]
        if end_cycle[0] > threshold :
            paths[0] = 1
        if end_cycle[1] > threshold :
            paths[1] = 1
        if end_cycle[2] > threshold :
            paths[2] = 1
            if cycle == -1 :
                cycle = c
        if end_cycle[3] > threshold :
            paths[3] = 1
            if cycle == -1 :
                cycle = c
    
    return t, v, scenario(paths), cycle




def color_code_scenario (s) :
    """
    Color code for observed evolution scenarios
    Arguments:
        s: number of observed scenario
    Returns:
        corresponding color
    """
    if s == -1 :
        return "lightgrey" # no mutant detected
    elif s == 0 :
        return "indigo" # mutant 10 detected
    elif s == 1 :
        return "skyblue" # mutants 10 and 11 (from 10) detected
    elif s == 2 :
        return "yellowgreen" # mutants 10 and 01 detected
    elif s == 3 :
        return "yellow" # mutants 10, 01 and 11 (from 10) detected
    elif s == 4 :
        return "darkorange" # mutants 10, 01 and 11 (from 01) detected
    elif s == 5 :
        return "red" # mutants 10, 01 and 11 (from both 10 and 01) detected
    



def color_code_cycle (s) :
    """
    Color code for observed cycle of double mutant emergence
    Arguments:
        s: cycle number
    Returns:
        corresponding color
    """
    if s == 1 :
        return "green"
    elif s == 2 :
        return "yellowgreen"
    elif s == 3 :
        return "gold"
    elif s == 4 :
        return "darkorange"
    elif s >= 5 :
        return "orangered"
    elif s == -1 :
        return "red"



    
def observed_scenario (s) :
    """
    Output to print for observed scenario
    Arguments:
        s: number of observed scenario
    Returns:
        corresponding output to print
    """
    if s == -1 or s==0 or s==2 :
        return "no adaptation"
    elif s == 1 or s==3 :
        return "adaptation from 10"
    elif s == 4 :
        return "adaptation from 01"
    elif s == 5 :
        return "adaptation from 10 and 01"  




def plot_sim (n, nb_cycles, alpha, beta, delta, rates, p = None, s = 0.1, sep_double = True, leg_loc = "lower left", file = None) :
    """
    Plot the subpopulation sizes as a function of time, for a bottlenecked
    population simulated during nb_cycles
    Arguments:
        n: n parameter (inverse of total mutation rate)
        nb_cycles: number of cycles (growth + bottleneck) to simulate
        alpha: parameter for final wild-type population size
        beta: parameter for initial wild-type population size
        delta: parameter for low mutation rate
        rates: list of intrinsic birth and death rates of the 5 subpopulations
        p: proportion of carrying capacity reached by the WT before dilution
        (set to None to make a simulation without density-dependence)
        s: mutants' advantage for resource exploitation (not used if p=None)
        sep_double: whether or not to plot separately the two double mutants
        subpopulations (from 10 and from 01)
        leg_loc: legend location (set to None for no legend)
        file: file to save the figure
    """
    sim = sim_pop(n, nb_cycles, alpha, beta, delta, rates, p = p, s = s, scenario_only = False)
    
    mu = [1/n,1/(n**delta)]
    r00 = rates[0][0]*(1 - sum(mu)) - rates[1][0]    
    if p :
        tn = np.log((n**(alpha-beta) - 1)/(1/p-1))/r00
    else :
        tn = np.log(n)*(alpha-beta)/r00
    
    plt.plot(sim[0], np.log(sim[1][:,0])/np.log(n), color="cornflowerblue")
    plt.plot(sim[0], -np.log(sim[1][:,1])/np.log(n), color="orange")  
    plt.plot(sim[0], -np.log(sim[1][:,2])/np.log(n), color="yellow") 
    
    if sep_double :
        plt.plot(sim[0], -np.log(sim[1][:,3])/np.log(n), color="hotpink")
        plt.plot(sim[0], -np.log(sim[1][:,4])/np.log(n), color="darkorchid")
        if leg_loc :
            plt.legend(["WT", "10 mutant", "01 mutant", "double mutant from 10", "double mutant from 01"], loc = leg_loc)
        plt.fill_between(sim[0], np.log(sim[1][:,0])/np.log(n), color='cornflowerblue', alpha=0.5)
        plt.fill_between(sim[0], -np.log(sim[1][:,1])/np.log(n), color='orange', alpha=0.5)
        plt.fill_between(sim[0], -np.log(sim[1][:,2])/np.log(n), color='yellow', alpha=0.5)
        plt.fill_between(sim[0], -np.log(sim[1][:,4])/np.log(n), color='darkorchid', alpha=0.5)
        plt.fill_between(sim[0], -np.log(sim[1][:,3])/np.log(n), color='hotpink', alpha=0.5) 
        
    else :
        plt.plot(sim[0], -np.log(sim[1][:,3]+sim[1][:,4])/np.log(n), color="hotpink")
        if leg_loc :
            plt.legend(["WT", "10 mutant", "01 mutant", "double mutants"], loc = leg_loc, frameon = False)
        plt.fill_between(sim[0], np.log(sim[1][:,0])/np.log(n), color='cornflowerblue', alpha=0.5)
        plt.fill_between(sim[0], -np.log(sim[1][:,1])/np.log(n), color='orange', alpha=0.5)
        plt.fill_between(sim[0], -np.log(sim[1][:,2])/np.log(n), color='yellow', alpha=0.5)
        plt.fill_between(sim[0], -np.log(sim[1][:,3]+sim[1][:,4])/np.log(n), color='hotpink', alpha=0.5)
        
    plt.title(r"$\alpha={}$".format(alpha)+r", $\beta={}$".format(beta)) 
    plt.yticks(np.arange(-2,2,0.5), np.concatenate((np.arange(2,0,-0.5), np.arange(0,2,0.5)))) 
    plt.xlabel("time (number of cycles)")
    plt.xticks([i*tn for i in range(nb_cycles+1)], range(nb_cycles+1))
    plt.ylabel(r"population size $(\log_n)$")
    if file :
        plt.savefig(file)
    plt.show()
    print("Observed scenario: ", observed_scenario(sim[2]))
    print("Observed cycle of double mutant emergence: cycle", sim[3])




def many_sim_pop (n_sim, n, nb_cycles, delta, rates, p = None, s = 0.1) :
    """
    Simulates the evolution of several bottlenecked population, with
    (uniformly) random alpha and beta parameters
    Arguments:
        n_sim: number of distinct populations to simulate
        n: n parameter (inverse of total mutation rate)
        nb_cycles: number of cycles (growth + bottleneck) to simulate
        delta: parameter for low mutation rate
        rates: list of intrinsic birth and death rates of the 5 subpopulations
        p: proportion of carrying capacity reached by the WT before dilution
        (set to None to make a simulation without density-dependence)
        s: mutants' advantage for resource exploitation (not used if p=None)
    Returns:
        A: list of alpha parameters
        B: list of beta parameters
        S: list of observed scenarios
        C: list of observed cycle of emergence for double mutants
    """
    A = []
    B = []
    S = []     
    C = []
    
    for i in range(n_sim) :
        if i%10 == 0 :
                print(i)
        alpha = np.random.uniform(low = 1.01, high = 1.99)
        beta = np.random.uniform(low = 0.01, high = 0.99)
        
        try :
            if (not p) and n>=10**9 and alpha>1.4 :
                sim = sim_pop(n, 4, alpha, beta, delta, rates, p = p, s = s, scenario_only = True)
            sim = sim_pop(n, nb_cycles, alpha, beta, delta, rates, p = p, s = s, scenario_only = True)
            A.append(alpha)
            B.append(beta)
            S.append(sim[2])
            C.append(sim[3])
        except :
            print(alpha,beta)
    
    return A, B, S, C




def make_fig_paths (A, B, S, predictions = False, rates = None, delta = None, file = None) :
    """
    Plot color points corresponding to observed scenarios in simulated populations, as a function of beta and alpha
    Arguments:
        A: list of alpha parameters used for simulations (e.g. output of
        many_sim_pop)
        B: list of beta parameters used for simulations (e.g. output of
        many_sim_pop)
        S: list of observed scenarios during simulations (e.g. output of
        many_sim_pop)
        predictions: wheter or not to plot threshold lines for large n
        predictions
        rates: list of intrinsic birth and death rates used for simulations (a
        value is required to plot predictions)
        delta: parameter for low mutation rate used for simulations (a value is
        required to plot predictions)
        file: file to save the figure
    """
    plt.figure(figsize=(8, 8))
    plt.scatter(B, A, c = [color_code_scenario(s) for s in S], alpha = 0.7)
    
    if predictions :
        b = np.arange(0, 1.01, 0.01)
        r00 = rates[0][0]-rates[1][0]
        r10 = rates[0][1]-rates[1][1]
        r01 = rates[0][2]-rates[1][2]
        plt.plot(b, [delta for x in b], 'k--') # emergence of strongly beneficial mutants
        plt.plot(b, [1+delta*r00/r10 for x in b], 'k--') # 10->11 at first cycle
        plt.plot(b, [delta+r00/r01 for x in b], 'k--', label = "") # 01->11 at first cycle
        plt.plot(b, [(r10/r00-x)/(r10/r00-1) for x in b], 'k--') # survival of weakly beneficial mutants
        plt.plot(b, [(delta*r01/r00-x)/(r01/r00-1) for x in b], 'k--') # survival of strongly beneficial mutants
    
    plt.xlim(0,1)
    plt.xticks(np.arange(0,1.1,0.2)) 
    plt.ylim(1,2)
    plt.yticks(np.arange(1,2.1,0.2)) 
    if file :
        plt.savefig(file)
    plt.show()




def make_fig_double (A, B, C, predictions = False, rates = None, delta = None, file = None) :
    """
    Plot color points corresponding to cycle of double mutant emergence in
    simulated populations, as a function of beta and alpha
    Arguments:
        A: list of alpha parameters used for simulations (e.g. output of
        many_sim_pop)
        B: list of beta parameters used for simulations (e.g. output of
        many_sim_pop)
        C: list of observed cycle of emergence during simulations (e.g. output
        of many_sim_pop)
        predictions: wheter or not to plot threshold lines for large n
        predictions
        rates: list of intrinsic birth and death rates used for simulations (a
        value is required to plot predictions)
        delta: parameter for low mutation rate used for simulations (a value is
        required to plot predictions)
        file: file to save the figure
    """
    plt.figure(figsize=(8, 8))
    plt.scatter(B, A, c = [color_code_cycle(s) for s in C], alpha = 0.7)
    
    if predictions :
        b = np.arange(0, 1.01, 0.01)
        r00 = rates[0][0]-rates[1][0]
        r10 = rates[0][1]-rates[1][1]
        r01 = rates[0][2]-rates[1][2]
        plt.plot(b, [delta for x in b], 'k--') # emergence of strongly beneficial mutants
        plt.plot(b, [1+delta*r00/r10 for x in b], 'k--') # 10->11 at first cycle
        plt.plot(b, [delta+r00/r01 for x in b], 'k--', label = "") # 01->11 at first cycle
        plt.plot(b, [(r10/r00-x)/(r10/r00-1) for x in b], 'k--') # survival of weakly beneficial mutants
        plt.plot(b, [(delta*r01/r00-x)/(r01/r00-1) for x in b], 'k--') # survival of strongly beneficial mutants
        for k in range(2,5) : # emergence of double mutants from 10 at cycle k
            plt.plot(b, [(r10/r00+delta+x*(k-1)*(r10/r00-1))/(r10/r00+(k-1)*(r10/r00-1)) for x in b], 'k--')
    
    plt.xlim(0,1)
    plt.xticks(np.arange(0,1.1,0.2)) 
    plt.ylim(1,2)
    plt.yticks(np.arange(1,2.1,0.2)) 
    if file :
        plt.savefig(file)
    plt.show()





