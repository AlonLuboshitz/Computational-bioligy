import numpy as np
import pandas as pd
import random
import itertools
from itertools import combinations
from matplotlib import pyplot as plt
import sys



'''
Function to create cost matrix from priorities (1-based) input.
cost defiend as the sum of:
    * the priority of man i considering woman j
    * the priority of women j considering man i
'''
def create_cost_matrix(priorities_input, population_size):
    # seperate into men and women prioriies
    men_priorities = priorities_input.iloc[:population_size, :]
    women_priorities = priorities_input.iloc[population_size:, :]

    # calculate cost for each combination of men and women
    for i in range(population_size):
        for j in range(population_size):
            cost_matrix[i, j] = men_priorities.iloc[i, j] + women_priorities.iloc[j, i]


'''
function to init solutions matrix.
Create n (num_of_solutions) random arrays with p (population_size) cells.
the cell j represent the paring of men j with the women defiend in the value of the cell
----
for example:
|2|4|3|1|
cell 0 represent the pairing of man 0 with woman 2
'''
def init_solotions(population_size, num_of_solutions):
    initial_sol = np.zeros((num_of_solutions,population_size))

    for i in range(num_of_solutions):
        initial_sol[i] = np.array(random.sample(range(population_size), population_size))
    
    return initial_sol.astype(int)

'''
 function to calculte vector with the cost for each solution.
 cost calcultes as the sum of the cost (fron the cost matrix) of each pair
 '''
def cost_function(sol_matrix):
    # extract number of solutions and population size
    num_of_sol = sol_matrix.shape[0]
    population = sol_matrix.shape[1]

    # init parameters
    cost_vector = np.zeros(num_of_sol)
    sum = 0

    for i in range(num_of_sol): # for each solutions
        for j in range(population):
            # extract the pairing
            men_index = j
            women_index = sol_matrix[i][j]
            # add the cost of the pairing 
            sum += cost_matrix[men_index][women_index]
        cost_vector[i] = sum
        sum = 0

    return cost_vector.astype(int)

'''
function to add the min,man,avg values of the cost vetor of the gen_num generation.
'''
def cost_eval_per_gen(gen_num, cost_vec,generations_costs):
    generations_costs[gen_num] = [cost_vec.min(), cost_vec.max(), cost_vec.mean()]

"""
Mutate the solution vector by performing a specified number of swaps.
"""
def mutate_solution(sol_vec, num_of_mutations = 1):
    population = sol_vec.shape[0]

    for _ in range(num_of_mutations):
        # Get two random indices to swap
        indices = random.sample(range(population), 2)
        index_1 = indices[0]
        index_2 = indices[1]

        # Swap the elements at the selected indices
        sol_vec[index_1], sol_vec[index_2] = sol_vec[index_2], sol_vec[index_1]

    return sol_vec

# Function to fix duplicates in the vectors
def fix_duplicates(sol_vec, length):
    unique_elements = set()
    duplicates = []
        
    # Identify duplicates and unique elements
    for i in range(length):
        if sol_vec[i] in unique_elements:
            duplicates.append(i)
        else:
            unique_elements.add(sol_vec[i])
        
    # Identify missing elements
    missing_elements = set(range(length)) - unique_elements
        
    # Replace duplicates with missing elements
    for i in duplicates:
        sol_vec[i] = missing_elements.pop()
        
    return sol_vec

'''
cross over the 2 sultion at random index, validate (and fix of needed) each ne vector containes only uniqly numbers
'''
def cross_over(sol_vec_1, sol_vec_2):
    length = sol_vec_1.shape[0]
    crossover_point = random.randint(1, length - 1)
                                     
    # Create the new vectors by crossing over
    new_sol_vec_1 = np.concatenate((sol_vec_1[:crossover_point], sol_vec_2[crossover_point:]))
    new_sol_vec_2 = np.concatenate((sol_vec_2[:crossover_point], sol_vec_1[crossover_point:]))

    # fix duploclications (make sure all values are unique)
    new_sol_vec_1  = fix_duplicates(new_sol_vec_1, length)
    new_sol_vec_2  = fix_duplicates(new_sol_vec_2, length)

    return new_sol_vec_1, new_sol_vec_2


def genetic_algo(population_size,num_of_solutions,generations, mut_perc,cross_perc,best_perc,num_of_mutations):
    generations_costs = np.zeros((generations,3))
    create_cost_matrix(GA_input,population_size)
    solutions = init_solotions(population_size, num_of_solutions)
    

    # Calculate the number of solutions for each "alteration"
    num_best = int((best_perc/100) * num_of_solutions)
    num_mutate = int((mut_perc/100) * num_of_solutions)
    num_cross = int((cross_perc/100) * num_of_solutions)
    if((num_cross % 2) != 0):
            num_cross+=1
    num_untouched = num_of_solutions - num_best - num_mutate - num_cross
    
    for gen in range(generations):
        # evaluate and keep cost of current solutions
        cost_vec = cost_function(solutions)
        cost_eval_per_gen(gen,cost_vec,generations_costs)

        # Sort solutions based on cost (accending: lower cost is better)
        sorted_indices = np.argsort(cost_vec)
        sorted_solutions = solutions[sorted_indices]

        new_solutions = []

        # Keep the best solutions
        new_solutions.extend(sorted_solutions[:num_best])

        # mutate random solution and store it in new solutions
        for i in range(num_mutate):
            rand_index = random.choice(range(num_of_solutions))
            mutated_sol = mutate_solution(solutions[rand_index].copy(), num_of_mutations)
            new_solutions.append(mutated_sol)

        # cross- over
        for i in range(num_cross //2):
            indices = random.sample(range(num_of_solutions), 2)
            index_1 = indices[0]
            index_2 = indices[1]
            new_sol_1, new_sol_2 = cross_over(solutions[index_1].copy(), solutions[index_2].copy())
            new_solutions.extend([new_sol_1, new_sol_2])
        
        # add "as is" random sol to fil matrix
        for i in range(num_untouched):
            rand_index = random.choice(range(num_of_solutions))
            new_solutions.append(solutions[rand_index])

        assert(len(new_solutions) == len(solutions))

        solutions = np.array(new_solutions)

        # if (gen % 100 == 0):
        #     print(f"gen {gen}, cost: {generations_costs[gen]}")
        # print(f"gen: {gen}, solutions:\n {solutions}\n")

    return [generations-1][0], sorted_solutions[0]

    #print(generations_costs)
def parse_gen_algo_output(min_cost, best_solution):
    worst_cost = couples_population_size * (couples_population_size - 1) * 2
    score = ((worst_cost - min_cost) / worst_cost) * 100
    best_solution = best_solution + 1
    print(f"Best solution: {best_solution}")
    print(f"Best score: {score:.2f} %")

def grid_search(combinations):
    results = []
    for comb in combinations:
        generations_costs = genetic_algo(couples_population_size, *comb)
        cost = generations_costs[comb[1] -1]
        results.append((comb, cost))
        # print(f" num_of_solutions: {num_of_solutions},generations: {generations}, mut_perc: {mut_perc},cross_perc: {cross_perc},best_perc: {best_perc},num_of_mutations: {num_of_mutations}\ncost: {generations_costs[generations-1]}\n ----------------------------------------\n")
    return results

def run_grid_search():
    # create solutions number range and generation number range that gen*sol =18k
    num_of_solutions = range(100, 201, 10)
    generations = range(100,201,10)
    sol_and_gen_combinations = list(itertools.product(num_of_solutions,generations))
    sol_range_gen_range = []
    for comb in sol_and_gen_combinations: 
        if ((comb[0] * comb[1]) == 18000):
            sol_range_gen_range.append((comb[0],comb[1]))
            
    # create all combinations of the parameters
    mut_perc = range(1, 31, 10)
    cross_perc = range(1, 31, 10)
    best_perc = range(1, 31, 10)
    num_of_mutations = range(1,4,1)
    all_combinations = list(itertools.product(sol_range_gen_range,mut_perc,cross_perc,best_perc,num_of_mutations))
    all_combinations = [(comb[0][0],comb[0][1],comb[1],comb[2],comb[3],comb[4]) for comb in all_combinations]
    results = grid_search(all_combinations)
    results.sort(key=lambda x: x[1][0])
    best_50_combinations = results[:50]
    
    
    # save results to file
    solution_series = [comb[0] for comb,cost in best_50_combinations]
    gen_series = [comb[1] for comb,cost in best_50_combinations]
    mut_perc_series = [comb[2] for comb,cost in best_50_combinations]
    cross_perc_series = [comb[3] for comb,cost in best_50_combinations]
    best_perc_series = [comb[4] for comb,cost in best_50_combinations]
    num_of_mutations_series = [comb[5] for comb,cost in best_50_combinations]
    results_df = pd.DataFrame({'num_of_solutions': solution_series, 'generations': gen_series, 'mut_perc': mut_perc_series, 'cross_perc': cross_perc_series, 'best_perc': best_perc_series, 'num_of_mutations': num_of_mutations_series })
    results_df.to_csv("results.txt",index=False)

def plot_score(ax,costs,sultions_number,generations,mut_prec,cross_perc,best_perc,num_of_mutations):
    '''This function plot the costs generated from a spesific run.
    Args:
    1. costs = the min,avg,max costs of each generation
    2. params = the parameters of the run - generation number,solution number,mutation percentage,cross over percentage,best percentage,number of mutations
    ----------
    returns a sub plot of the costs
    '''
    generations_range = range(1,generations+1)
    mins = [cost[0] for cost in costs]
    maxs = [cost[1] for cost in costs]
    avgs = [cost[2] for cost in costs]
    ax.scatter(generations_range, mins, color='blue', label='min')
    
    ax.scatter(generations_range, avgs, color='green', label='average')
    ax.scatter(generations_range, maxs, color='red', label='maximum')
    ax.plot(generations_range, mins, color='blue')
    ax.plot(generations_range, avgs, color='green')
    ax.plot(generations_range, maxs, color='red')

    # Set title and labels
    ax.set_title(f'Results for {sultions_number} solutions with {generations} generations\n mutation percentage: {(mut_prec-1)}%, cross over percentage: {cross_perc-1}%, best percentage: {best_perc-1}%, number of mutations: {num_of_mutations}',fontsize=6)
    ax.set_xlabel('generation number')
    ax.set_ylabel('Scores')
    return ax



def main_plot(values):
    # Create a figure with subplots
    cols =int(len(values)/2) 
    fig, axs = plt.subplots(2, cols, figsize=(12, 10))
    # Create scatter plots in separate functions
    for i, (params, costs) in enumerate(values):
        plot_score(axs[i // (len(values) // 2), i % (len(values) // 2)], costs, *params)
    
    # Set common x and y labels
    fig.text(0.5, 0.04, 'generation number', ha='center', va='center', fontsize=12)
    fig.text(0.06, 0.5, 'Scores', ha='center', va='center', rotation='vertical', fontsize=12)

    # Create a single legend for the entire figure
    handles, labels = axs[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper right')
     # Adjust layout
    plt.tight_layout()
    
    # Show the plot
    plt.show()
    
    # Save the plot
    fig.savefig("scatter_plots.png")

def plot_best_combinations():
    results_df = pd.read_csv("results.txt")
    # find first indexes of each unique number of solutions
    unique_solutions = results_df['num_of_solutions'].unique()
    first_indexes = []
    for solution in unique_solutions:
        first_index = results_df.index[results_df['num_of_solutions'] == solution][0]
        first_indexes.append(first_index)
    values = []
    # Get sol_num,gen_num,mut_perc,cross_perc,best_perc,num_of_mutations for each unique number of solutions
    for index in first_indexes:
        params_tuple = (results_df['num_of_solutions'][index], results_df['generations'][index], results_df['mut_perc'][index], results_df['cross_perc'][index], results_df['best_perc'][index], results_df['num_of_mutations'][index])
        cost_per_run = genetic_algo(couples_population_size,*params_tuple)
        values.append((params_tuple, cost_per_run))
    # Plots results
    main_plot(values) 




def init_input(path):
    global couples_population_size,cost_matrix, GA_input
    GA_input = pd.read_csv(path, sep=' ', header=None)
    GA_input = GA_input.applymap(lambda x : (x-1) )
    couples_population_size = GA_input.shape[1] # =30

    # init paremetres
    cost_matrix = np.zeros((couples_population_size,couples_population_size))
if __name__ == "__main__":
    init_input(sys.argv[1])
    #run_grid_search()
    #run_genetic_algo()
    cost,sul = genetic_algo(couples_population_size,100,180,10,10,10,1)
    parse_gen_algo_output(cost,sul)
