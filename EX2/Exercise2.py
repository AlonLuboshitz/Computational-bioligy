import numpy as np
import pandas as pd
import random
import itertools
from itertools import combinations

# read input
input_path = "/home/gili/Computational-bioligy/EX2/GA_input.txt"
GA_input = pd.read_csv(input_path, sep=' ', header=None)
GA_input = GA_input.map(lambda x : (x-1) )
print(GA_input)
couples_population_size = GA_input.shape[1] # =30

# init paremetres
cost_matrix = np.zeros((couples_population_size,couples_population_size))
generations = 2000

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

    return generations_costs
    #print(generations_costs)


def grid_search(combinations):
    results = []
    for comb in combinations:
        generations_costs = genetic_algo(couples_population_size, *comb)
        cost = generations_costs[comb[1] -1]
        results.append((comb, cost))
        # print(f" num_of_solutions: {num_of_solutions},generations: {generations}, mut_perc: {mut_perc},cross_perc: {cross_perc},best_perc: {best_perc},num_of_mutations: {num_of_mutations}\ncost: {generations_costs[generations-1]}\n ----------------------------------------\n")
    return results

num_of_solutions = range(100, 201, 10)
generations = range(100,201,10)

sol_and_gen_combinations = list(itertools.product(num_of_solutions,generations))
sol_range = []
gen_range = []
for comb in sol_and_gen_combinations:
    if ((comb[0] * comb[1]) == 18000):
        sol_range.append(comb[0])
        gen_range.append(comb[1])


mut_perc = range(1, 16, 5)
cross_perc = range(1, 16, 5)
best_perc = range(1, 16, 5)
num_of_mutations = range(1,4,1)

all_combinations = list(itertools.product(sol_range, gen_range,mut_perc,cross_perc,best_perc,num_of_mutations))

results = grid_search(all_combinations[100:120])
print(results)
results.sort(key=lambda x: x[1][0])
    
# Get top 10 combinations
best_10_combinations = results[:10]
    
for comb, cost in best_10_combinations:
        print(f"Best Combination -> num_of_solutions: {comb[0]}, generations: {comb[1]}, mut_perc: {comb[2]}, cross_perc: {comb[3]}, best_perc: {comb[4]}, num_of_mutations: {comb[5]}\ncost: {cost}\n ----------------------------------------\n")



