# Computational-bioligy

1. Function to create cost matrix from priority V
2. Create function to init solutions string (number of solutions -s, population_size-n):
Create s random arrays with n cells each cell represents the I male, and value is the J female.
(random without replacement (1-n)) - make sure no repeats. V
3. Cost function (matrix,sultions) return vector n size wtih costs of each solution V
4. Cost evalaution: keep global cost vector(size generation*(min,man,avg)) V
5. Mutation function(sultion) - switch 2 cells in the vector. V
6. Cross-over mutation (sul1,sul2) - cross over the 2 sultion at random index, and correct output and return 2 sultions. V
7. Iterative function (n,generations, mut_%,cross_%,best_%) :
  init sultions
  for each generation:
keep best %
complete: cross% mut%
and fill gap with random from untouched solutions
** make sure sultions size stays the same
   Evaluate cost into global array
8. Each 5 generations, plot the costs.

** Local minimum? (converges of populations) - after i generations if min~=max~=average pop converge:
insert mutations (Keep best sultions tho)

early stopping: if min = 0 stop YOU WON GILI 


  
10. Evulate cost function
