import networkx as nx
import numpy as np
import math
import matplotlib.pyplot as plt
import pandas as pd
from IPython.display import display
%matplotlib inline

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import nevergrad as ng
import cma

from IPython.display import Markdown
def print_markdown(string):
    display(Markdown(string))
def printmd(string):
    display(Markdown(string))
    
    
# Simulates an epidemic on an unweighted graph whith adjacency matrix W.
# Diseas propagation model is a discrete-time version of the SIR epidemic model.
# 'A': dictionary with possible states.
# 'beta': probability that the infection is spread from an infected individual to a susceptible one (given that they are connected by a link) during one time step.
# 'ro': probability that an infected individual will recover during one time step.
# 'vacc_percentage': percentage of vaccinated individuals, i-th element represents the percentage of vaccinated individuals by time unit i. It makes sense only if 'vaccination' is set to True.
# 'n_infected_initial': number of infected nodes in initial configuration.
# 'n_steps': time units for which the simulation will go on.
# 'initial_configuration': method to generate initial configuaration. Possible choices = ['random','equally spaced','all grouped','equally spaced groups:n'] with n number of groups
# 'vaccination': determines if the epidemics has to be simulated with vaccination (True) or not (False).

# It returns a matrix such that:
# each row i represents the state of each agent at time i.
# each column j represents the state of agent j.

def simulate(A, W, beta, ro, n_infected_initial = 10, n_steps = 15, initial_configuration = 'random', vacc_percentage = None, vaccination = False):
    
    n_steps += 1 #(week 0: initial situation)
    n_agents = W.shape[0]
    
    if vaccination:
        vacc_percentage_perWeek = np.diff(vacc_percentage)/100
    
    # store the ids of visited configurations, describing the state of each agent
    states = np.full((n_steps, n_agents), A['susceptible'], dtype=int)
    # in the initial configuration, n_infected_initial random agents are infected
    x0 = np.full(n_agents, A['susceptible'])
    if initial_configuration == 'random':
        x0[np.random.choice(n_agents, n_infected_initial, replace=False)] = A['infected'] # replace=False to avoid the same node be chosen twice
    if initial_configuration == 'equally spaced':
        rate = round(n_agents / n_infected_initial) # number of elements between 2 infected nodes (to be precise: rate = (number of elements between 2 infected nodes) + 1)
        inf_node = np.random.choice(rate, 1)[0] # infected node between nodes (0,...rate-1)
        i = inf_node
        while i < n_agents:
            x0[i] = A['infected']
            i+=rate
    if initial_configuration == 'all grouped':
        inf_node = np.random.choice(n_agents, 1)[0] # infected node in the center of the group
        x0[inf_node] = A['infected']
        n_inf = 1
        i = 1
        up = True
        while(n_inf<n_infected_initial):
            if up:
                x0[(inf_node+i)%n_agents] = A['infected']
                up = False
            else:
                x0[(inf_node-i)%n_agents] = A['infected']
                up = True
                i+=1
            n_inf+=1
    if initial_configuration.split(":")[0] == 'equally spaced groups':
        n_groups = int(initial_configuration.split(":")[1])
        nodes_per_group = [a.shape[0] for a in np.array_split(range(n_infected_initial), n_groups)] # number of nodes in each group
        rate = round(n_agents / n_groups) # number of elements between 2 infected groups of nodes (their central nodes) (to be precise: rate = (number of elements between 2 infected nodes) + 1)
        inf_node = np.random.choice(n_agents, 1)[0] # central infected node between nodes (0,...n_agents-1) in the first group
        i = inf_node
        tot_inf = 0 # infected nodes counter
        gr = 0 # groups counter
        while tot_inf < n_infected_initial:
            x0[i] = A['infected']
            tot_inf+=1
            n_inf = 1 # infected nodes of group 'gr' counter
            j = 1
            up = True
            while(n_inf<nodes_per_group[gr] and tot_inf<n_infected_initial):
                if up:
                    x0[(i+j)%n_agents] = A['infected']
                    tot_inf+=1
                    up = False
                else:
                    x0[(i-j)%n_agents] = A['infected']
                    tot_inf+=1
                    up = True
                    j+=1
                n_inf+=1
            i+=rate
            i=i%n_agents
            gr+=1
    # define initial state id (time 0)
    states[0] = x0

    if vaccination:
        # administrate vaccinations
        vacc_ind = np.random.choice(np.where(is_notVaccinated(states[0]))[0], round(vacc_percentage_perWeek[0]*n_agents), replace=False)
        states[0, vacc_ind] = administer_vaccinate(states[0, vacc_ind])
    
    # for each step of the simulation
    for i in range(1,n_steps): # for each unit of time
        for j in range(n_agents): # for each agent
            if states[i-1,j] == A['susceptible']: # if j-th agent is 'susceptible'
                m = np.sum(states[i-1][np.where(W[j]==1)[1]]==A['infected']) # calculate the number of infected neighbours of j-th agent
                probability = 1 - ((1-beta)**m) # calculate probability for j-th agent to get infected
                if np.random.rand() < probability:
                    states[i,j] = A['infected'] # j-th agent get infected
                else:
                    states[i,j] = states[i-1,j] # j-th agent does not get infected
            elif is_infected(states[i-1,j]): # if j-th agent is 'infected' or 'infected but vaccinated'
                probability = ro # calculate probability for j-th agent to get recovered
                if np.random.rand() < probability:
                    states[i,j] = recover(states[i-1,j]) # j-th agent get recovered
                else:
                    states[i,j] = states[i-1,j] # j-th agent does not get recovered
            else: # if j-th agent is 'recovered' or 'susceptible but vaccinated' or 'recovered but vaccinated' 
                states[i,j] = states[i-1,j] # j-th agent does not modify its state
             
        if vaccination:
            # administrate vaccinations
            vacc_ind = np.random.choice(np.where(is_notVaccinated(states[i]))[0], round(vacc_percentage_perWeek[i]*n_agents), replace=False)
            states[i, vacc_ind] = administer_vaccinate(states[i, vacc_ind])
            
    return states
  
# Generates and returns a random graph of n nodes with average degree k>0 by using preferential attachment.
# if 'demo' is True, print the graph at timestep t.

def generate_gpa(n, k, demo=True):

    c = k/2

    t = 1
    G_1 = nx.complete_graph(k+1)

    if demo:
        warnings.simplefilter(action='ignore', category=RuntimeWarning)
        plt.figure(figsize=(20, 20))
        plt.title("G_" + str(t), fontsize=60)
        pos = nx.circular_layout(G_1)
        nx.draw(G_1, pos, node_color='#FFA07A', arrowsize=30, node_size=3000, font_size=40, font_color='white', with_labels=True)

    t_final = n-k
    up = False
    G = G_1
    for t in range(2, t_final+1):
        deg_id = np.array([n for n,d in list(G.degree())])
        deg_val = np.array([d for n,d in list(G.degree())])
        norm_deg_val = deg_val/np.sum(deg_val)

        if up:
            c_round = math.ceil(c)
            up = False
        else:
            c_round = math.floor(c)
            up = True

        neighbors = np.random.choice(deg_id, p=norm_deg_val, size=c_round, replace=False) # replace=False guarantees no neighbor is chosen twice
        G.add_node(k+t-1)
        for neigh in neighbors:
            G.add_edge(k+t-1,neigh) 

        if demo:
            if t<=20:
                plt.figure(figsize=(20,20))
                plt.title("G_" + str(t), fontsize=60)
                pos = nx.circular_layout(G)
                nx.draw(G, pos, node_color='#FFA07A', arrowsize=30, node_size=3000, font_size=40, font_color='white', with_labels=True)
                
    return G
  
def is_vaccinated(n):
    return n>=3

def is_notVaccinated(n):
    return n<3

def administer_vaccinate(n):
    return n+3

def is_susceptible(n):
    return n%3==0

def is_infected(n):
    return n%3==1

def is_recovered(n):
    return n%3==2

def recover(n):
    return n+1
  
def RMSE(true_value, computed_value):
    return np.sqrt(np.sum((computed_value-true_value)**2)/(true_value.shape[0]))

counter = 1

def objective_function(params, n_simulations, weeks, A, n_infected_initial, vacc, inf_0):
    k = params[0]
    beta = params[1]
    ro = params[2]
    
    G = generate_gpa(n, k, demo=False)
    n_agents = len(G)
    W = nx.convert_matrix.to_numpy_matrix(G)

    cum_newly_infected = np.zeros((n_simulations, weeks))
    A_rev = dict((v,k) for k,v in A.items())
    for i in range(n_simulations):
        states = simulate(A, W, beta, ro, n_infected_initial, weeks, vacc_percentage=vacc, vaccination=True)
        newly_infected = np.array([np.sum(np.logical_and(is_susceptible(states[i-1]), is_infected(states[i]))) for i in range(1, weeks+1)]) # newly infected agent are such that at time unit i-1 they were suscptible and at time unit i they are infected
        cum_newly_infected[i] = newly_infected
    avg_newly_infected = np.mean(cum_newly_infected,axis=0)

    err = RMSE(inf_0[1:], avg_newly_infected)
    
    print("Evaluation number " + str(count) + ". k = " + str(k) + ", beta = " + str(beta) + ", ro = " + str(ro) + ". RMSE = ", str(RMSE)) 
    
    counter+=1
    
    return err
  
n = 934
vacc = np.array([5, 9, 16, 24, 32, 40, 47, 54, 59, 60, 60, 60, 60, 60, 60, 60])
vacc = np.insert(vacc,0,0) # by week i<0 0% of individuals is vaccinated
inf_0 = np.array([1, 1, 3, 5, 9, 17, 32, 32, 17, 5, 2, 1, 0, 0, 0, 0])

n_simulations = 100


n_infected_initial = inf_0[0]
weeks = 15

A = {}
A['susceptible'] = 0
A['infected'] = 1
A['recovered'] = 2
A['susceptible but vaccinated'] = 3
A['infected but vaccinated'] = 4
A['recovered but vaccinated'] = 5

printmd("$\mathcal{A}$=")
for a in A:
    print(a + ": " + str(A[a]))
   
k0 = 10
beta0 = 0.5
ro0 = 0.5

k_ranges = (1,20)
beta_ranges = (0,1)
ro_ranges = (0,1)

budget = 1000



    
searchSpace = []

k = ng.p.Scalar().set_bounds(lower=k_ranges[0], upper=k_ranges[1])
searchSpace.append(k)
beta = ng.p.Scalar().set_bounds(lower=beta_ranges[0], upper=beta_ranges[1])
searchSpace.append(beta)
ro = ng.p.Scalar().set_bounds(lower=ro_ranges[0], upper=ro_ranges[1])
searchSpace.append(ro)

params = ng.p.Tuple(*searchSpace)

instrumentation = ng.p.Instrumentation(params=params, n_simulations=n_simulations, weeks=weeks, A=A, n_infected_initial=n_infected_initial, vacc=vacc, inf_0=inf_0)
cmaES_optimizer = ng.optimizers.CMA(parametrization=instrumentation, budget=budget)
recommendation = cmaES_optimizer.minimize(objective_function)

rec = recommendation.value[1]['params']
optimum =  objective_function(**recommendation.kwargs)

print("Optimum, RMESE =", optimum)
print("Optimal values:")
printmd("$k=" + str(rec[0]))
printmd("$\beta=" + str(rec[1]))
printmd("$\ro=" + str(rec[2]))
  
