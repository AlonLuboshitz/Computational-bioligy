import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
from matplotlib.patches import RegularPolygon



def transfrom_to_nd(matrix):
    '''Function to reshape the matrix from (n,n,d) to (n^2,d)
    Args:
    1. matrix: np.array (n,n,d)
    -----------
    Returns: np.array (n^2,d)'''
    n = matrix.shape[0] * matrix.shape[1]
    A_reshaped = matrix.reshape(n, -1)
    return A_reshaped
def get_closest_neuron(x, matrix):
    '''Function iterate the matrix and return the neuron closest to the input x
    Args:
    1. x: input vector
    2. matrix: matrix of neurons (each row is a neuron, each column is a feature)
  
    -----------
    Returns:
    The flat index of the closest neuron'''
    
    if len(x) != matrix.shape[1]:
        raise ValueError('Input vector x has wrong dimension')
    # Calculate the Euclidean distances between x and each row of A_reshaped
    distances = np.linalg.norm(matrix - x, axis=1)
    # Find the flat index of the row with the minimum distance
    min_flat_index = np.argmin(distances)
    
    return min_flat_index

def init_x_prototypes(X, n_neurons):
    '''Function to init the neruons labels for the x data
    Args:
    1. X: input data (n_samples, d_features)
    2. n_neurons: number of neurons
    -----------
    Returns: x_nerons np.array (n_samples,) with the neuron labels'''
    return np.random.randint(0, n_neurons, size=X.shape[0])

def calculate_distances(X, x_neurons, matrix):
    ''''The function calculates the distance between each input vector and the neuron'''
    n_sqrt = matrix.shape[0]
    matrix = transfrom_to_nd(matrix)
    for x_index, x_vector in enumerate(X):
        min_neuron_index = get_closest_neuron(x_vector, matrix, n_sqrt)
        x_neurons[x_index] = min_neuron_index
    return x_neurons
class HexagonalMatrix:
    def __init__(self, rows, cols, pixels):
        self.rows = rows
        self.cols = cols
        self.pixels = pixels
        self.time_passed = 0
        self.matrix = np.zeros((rows, cols,pixels))
    
    def fit_x(self,X, x_prototypes, max_iter = 100):
        '''Function to "train" the matrix to fit the input data.
        Args:
        1. X: np.array (n_samples, d_features)
        2. x_prototypes: np.array (n_samples,) with the prototypes indices for each input.
        3. max_iter: int, number of iterations
        The function: 
        - init the matrix
        - transform it into (n^2,d)
        '''
        self.init_matrix()
        # Do self.init_matrix(X) for initating the matrix with values from X
        
        iteration = 0
        while self.time_passed < 180 and iteration < max_iter:
            for x_index, x_vector in enumerate(X):
                start_time = time.time()
                reshaped_matrix = transfrom_to_nd(self.matrix)
                min_neuron_index = get_closest_neuron(x_vector, reshaped_matrix)
                x_prototypes[x_index] = min_neuron_index
                i,j = np.unravel_index(min_neuron_index, (self.rows, self.cols))
                self.update_neighbors(i, j, x_vector)
                end_time = time.time()
                self.time_passed += end_time - start_time
            iteration += 1  
        self.time_passed = 0    
        return x_prototypes
        
    def init_matrix(self, X = None):
        '''Function init matrix randomly or by values sampled from X.
        Args:
        1. X - np.array (n_samples, d_features)
        -----------
        Sets: the self matrix with values drawn from X disterbution.'''
        if X is None:
            self.matrix = np.random.randint(0, 258, size=(self.rows, self.cols, self.pixels))
        else : pass ## Implement this part
        
    def get_matrix(self):
        return self.matrix
    def update_prototype(self, row, col, x_vector, learning_rate = 0.1):
        if 0 <= row < self.rows and 0 <= col < self.cols:
            distance = self.matrix[row, col] - x_vector
            self.matrix[row, col] = self.matrix[row, col] + learning_rate * distance
    
    def update_neighbors(self, row, col, x_vector):
        neighbors = self.get_neighbors(row, col)
        for r, c in neighbors:
            self.update_prototype(r, c, x_vector)
    
    def get_neighbors(self, row, col):
        if col % 2 == 0:
            directions = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1)]
        else:
            directions = [(-1, 0), (1, 0), (0, -1), (0, 1), (1, -1), (1, 1)]
            
        neighbors = []
        for dr, dc in directions:
            nr, nc = row + dr, col + dc
            if 0 <= nr < self.rows and 0 <= nc < self.cols:
                neighbors.append((nr, nc))
        return neighbors
    
    def display_matrix(self):
        fig, ax = plt.subplots(1, figsize=(10, 10))
        ax.set_aspect('equal')
        
        # Define hexagon size
        hex_size = 1
        for row in range(self.rows):
            for col in range(self.cols):
                # Calculate the center of each hexagon
                x = col * hex_size * 3/2
                y = row * np.sqrt(3) * hex_size + (col % 2) * (np.sqrt(3) / 2 * hex_size)
                hexagon = RegularPolygon((x, y), numVertices=6, radius=hex_size/np.sqrt(3),
                                         orientation=np.radians(30), edgecolor='k')
                ax.add_patch(hexagon)
                value = np.mean(self.matrix[row, col])
                ax.text(x, y, str(int(value)), ha='center', va='center', size=12)
        
        plt.xlim(-hex_size, self.cols * hex_size * 3/2)
        plt.ylim(-hex_size, self.rows * np.sqrt(3) * hex_size)
        plt.axis('off')
        plt.show()
def main(n_cols = 10,n_rows = 10):
    X = pd.read_csv("EX3/digits_test.csv")
    X = np.array(X)
    x_prototypes = init_x_prototypes(X, n_cols * n_rows)
    hex_matrix = HexagonalMatrix(n_rows,n_cols,X.shape[1])
    x_prototypes = hex_matrix.fit_x(X, x_prototypes)
    print(x_prototypes[:5])
if __name__ == "__main__":
    main()
    # Example usage


# print(x_neurons.shape,x_neurons[0])
