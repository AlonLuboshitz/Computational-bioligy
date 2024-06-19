import numpy as np
import pandas as pd
import random

# read input
input_path = "GA_input.txt"
GA_input = pd.read_csv(input_path, sep=' ', header=None)
couples_population_size = GA_input.shape[1] # =30

# init paremetres
cost_matrix = np.zeros((couples_population_size,couples_population_size))
generations = 50
generations_costs = np.zeros((generations,3))

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
def cost_eval_per_gen(gen_num, cost_vec):
    generations_costs[gen_num] = [cost_vec.min(), cost_vec.max(), cost_vec.mean()]

"""
Mutate the solution vector by performing a specified number of swaps.
"""
def mutate_solution(sol_vec, num_of_mutations):
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

def get_top_sol(cost_vec, best_perc):
    pass


def genetic_algo(population_size,num_of_solutions,generations, mut_perc,cross_perc,best_perc):
    create_cost_matrix(GA_input,population_size)
    solutions = init_solotions(population_size, num_of_solutions)

    for gen in range(generations):
        # evaluate and keep cost of current solutions
        cost_vec = cost_function(solutions)
        cost_eval_per_gen(gen,cost_vec)

        # Sort solutions based on cost (assuming lower cost is better)
        sorted_indices = np.argsort(cost_vec)
        sorted_solutions = solutions[sorted_indices]


        break

num_of_solutions = 10
genetic_algo(couples_population_size,num_of_solutions,generations, 5,5,5 )


