import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
from matplotlib.patches import RegularPolygon
from matplotlib.colors import ListedColormap
import seaborn as sns

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

    def fit_x(self,X, x_prototypes, sparse_perc = 0.8, levels=2, max_iter = 100, max_time=180):
        '''Function to "train" the matrix to fit the input data.
        Args:
        1. X: np.array (n_samples, d_features)
        2. x_prototypes: np.array (n_samples,) with the prototypes indices for each input.
        3. max_iter: int, number of iterations
        The function: 
        - init the matrix
        - transform it into (n^2,d)
        '''
        self.init_matrix(sparse_perc)
        # Do self.init_matrix(X) for initating the matrix with values from X
        
        iteration = 0
        while self.time_passed < max_time or iteration < max_iter:
            for x_index, x_vector in enumerate(X):
                start_time = time.time()
                reshaped_matrix = transfrom_to_nd(self.matrix)
                min_neuron_index = get_closest_neuron(x_vector, reshaped_matrix)
                x_prototypes[x_index] = min_neuron_index
                i,j = np.unravel_index(min_neuron_index, (self.rows, self.cols))

                learning_rates = [0.1,0.05,0.02]
                # update nuerons:
                self.update_prototype(i, j, x_vector, learning_rates[0])
                for level in range(1,levels+1):
                    self.update_neighbors(i, j, x_vector, level=level, learning_rate = learning_rates[level])
                
                # manage time handeling 
                end_time = time.time()
                self.time_passed += (end_time - start_time)
            iteration += 1  
        self.time_passed = 0    
        return x_prototypes
    
    def init_matrix(self,zeros_perc, X = None):
        '''Function init matrix randomly or by values sampled from X.
        Args:
        1. X - np.array (n_samples, d_features)
        -----------
        Sets: the self matrix with values drawn from X disterbution.'''
        total_elements = self.rows * self.cols * self.pixels
        n_zeros = int(zeros_perc * total_elements)
        n_random = total_elements - n_zeros

         # Create arrays
        zeros_array = np.zeros(n_zeros, dtype=int)
        random_array = np.random.randint(1, 258, size=n_random)  # Random ints between 1 and 257

        # Concatenate arrays
        combined_array = np.concatenate((zeros_array, random_array))

        # Shuffle the array to mix zeros and random integers
        np.random.shuffle(combined_array)

        # Reshape the array to the desired shape
        self.matrix = combined_array.reshape((self.rows, self.cols, self.pixels))

        
    def get_matrix(self):
        return self.matrix
    
    def update_prototype(self, row, col, x_vector, learning_rate = 0.1):
        if 0 <= row < self.rows and 0 <= col < self.cols:
            distance = (self.matrix[row, col] - x_vector)   
            self.matrix[row, col] = (self.matrix[row, col] + (learning_rate * distance))
    
    # def update_value(self, row, col, value):
    #     assert(value.shape[0] == self.pixels)
    #     if 0 <= row < self.rows and 0 <= col < self.cols:
    #         self.matrix[row, col] = value
    
    def update_neighbors(self, row, col, x_vector, level =1, learning_rate = 0.1):
        neighbors = self.get_neighbors(row, col, level)
        for r, c in neighbors:
            self.update_prototype(r, c, x_vector, learning_rate)
    
    def get_neighbors(self, row, col, level):
        original_nueron = set([(row, col)])
        def get_directions(col):
            if col % 2 == 0:
                return [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1)]
            else:
                return [(-1, 0), (1, 0), (0, -1), (0, 1), (1, -1), (1, 1)]
            
            
        directions = get_directions(col)
        first_neighbors = [(row + dr, col + dc) for dr, dc in directions if 0 <= row + dr < self.rows and 0 <= col + dc < self.cols]

        if level == 1:
            return first_neighbors
        
        neighbors = {1: first_neighbors}
        old_neighbors = set(first_neighbors)
        
        for current_level in range(2, level + 1):
            level_neighbors_set = {(neigh_row + dr, neigh_col + dc) for neigh_row, neigh_col in neighbors[current_level - 1] for dr, dc in get_directions(neigh_col) if 0 <= neigh_row + dr < self.rows and 0 <= neigh_col + dc < self.cols}
            neighbors[current_level] = level_neighbors_set
            if current_level != level:
                old_neighbors.update(level_neighbors_set)

        return neighbors[level] - old_neighbors - original_nueron


        
        # # first level neighbors
        # first_neighbors = []
        # directions = get_directions(col)
        # for dr, dc in directions:
        #     nr, nc = row + dr, col + dc
        #     if 0 <= nr < self.rows and 0 <= nc < self.cols:
        #         first_neighbors.append((nr, nc))
        # if (level == 1):
        #     return first_neighbors
    
        # neighbors = {}
        # neighbors[1] = first_neighbors
        # old_neighbors = set(first_neighbors)
 
        # for current_level in range(2, level +1):
        #     level_neighbors_set = set()
        #     for neigh_row, neigh_col in neighbors[current_level - 1]:
        #         directions = get_directions(neigh_col)
        #         for dr, dc in directions:
        #             nr, nc = neigh_row + dr, neigh_col + dc
        #             if 0 <= nr < self.rows and 0 <= nc < self.cols:
        #                 level_neighbors_set.add((nr, nc))
        #     neighbors[current_level] = level_neighbors_set
        #     if(current_level != level):
        #         old_neighbors.update(level_neighbors_set)
        # final_neigbors = neighbors[level] - old_neighbors
        # return final_neigbors
    
    def display_matrix(self, x_prototypes, label_df):
        fig, ax = plt.subplots(1, figsize=(10, 10))
        ax.set_aspect('equal')
        
        # Get unique labels and assign colors
        unique_labels = label_df['label'].unique()
        color_palette = sns.color_palette("hsv", len(unique_labels))
        label_to_color = {label: color for label, color in zip(unique_labels, color_palette)}

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
                flattened_index = ((row * self.rows) + col)
                value = get_neuron_label(flattened_index,x_prototypes, label_df)
                if value is not None:
                    most_frequent_label, percentage = value
                    # Adjust color based on percentage
                    base_color = label_to_color[most_frequent_label]
                    alpha = percentage / 100  # Scale percentage to [0, 1] for alpha
                    rgba_color = (*base_color, alpha)
                    ax.text(x, y, f'{most_frequent_label}\n{percentage:.1f}%', ha='center', va='center', size=6)
                    hexagon.set_facecolor(rgba_color)
                else:
                    hexagon.set_facecolor('gray')  # Default color for empty cells
        
        plt.xlim(-hex_size, self.cols * hex_size * 3/2)
        plt.ylim(-hex_size, self.rows * np.sqrt(3) * hex_size)
        plt.axis('off')
        plt.show()

def get_neuron_label(nueron_index, prototypes, labels):
    prototypes = pd.DataFrame(prototypes, columns=["prototypes"])
    filtered_prototypes  = prototypes.loc[prototypes['prototypes'] == nueron_index]
    neuron_labels = labels.loc[filtered_prototypes.index, 'label']

    if not neuron_labels.empty:
            most_frequent_label = neuron_labels.mode().iloc[0]
            count_of_most_frequent_label = (neuron_labels == most_frequent_label).sum()
            total_num_of_labels = len(neuron_labels)
            percentage = (count_of_most_frequent_label / total_num_of_labels) * 100 if total_num_of_labels != 0 else 0
            return most_frequent_label, percentage
    else:
        return None



def main(n_cols = 10,n_rows = 10):
    X = pd.read_csv("/home/gili/Computational-bioligy/EX3/digits_test.csv")
    X = np.array(X)

    x_prototypes = init_x_prototypes(X, n_cols * n_rows)
    hex_matrix = HexagonalMatrix(n_rows,n_cols,784)
    label_df = pd.read_csv("/home/gili/Computational-bioligy/EX3/digits_keys.csv", header=None, names=["label"]) # labels
    x_prototypes = hex_matrix.fit_x(X, x_prototypes,sparse_perc =0.8, levels=2, max_iter=10, max_time=20)
    hex_matrix.display_matrix(x_prototypes,label_df)


if __name__ == "__main__":
    main()

