import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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
def get_closest_neuron(x, matrix, n_sqrt):
    '''Function iterate the matrix and return the neuron closest to the input x
    Args:
    1. x: input vector
    2. matrix: matrix of neurons (each row is a neuron, each column is a feature)
    3. n_sqrt: number of rows/cols in the original matrix
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

def init_x_neurons(X, n_neurons):
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
        self.matrix = np.zeros((rows, cols,pixels))
    def get_matrix(self):
        return self.matrix
    def update_value(self, row, col, value):
        if 0 <= row < self.rows and 0 <= col < self.cols:
            self.matrix[row, col] = value
    
    def update_neighbors(self, row, col, value):
        neighbors = self.get_neighbors(row, col)
        for r, c in neighbors:
            self.update_value(r, c, value)
    
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

# Example usage
hex_matrix = HexagonalMatrix(5, 5,5)
hex_matrix.update_value(2,2, np.array([1,3,3,3,1]))
#hex_matrix.update_neighbors(2, 2, 2)
hex_matrix.display_matrix()
# X = pd.read_csv("EX3/digits_test.csv")
# X = np.array(X)
# x_neurons = init_x_neurons(X, 100)
# print(x_neurons.shape,x_neurons[0])
