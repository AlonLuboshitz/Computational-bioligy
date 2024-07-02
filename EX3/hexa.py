import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import RegularPolygon

class HexagonalMatrix:
    def __init__(self, rows, cols):
        self.rows = rows
        self.cols = cols
        self.matrix = np.zeros((rows, cols))
    
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
                value = self.matrix[row, col]
                ax.text(x, y, str(int(value)), ha='center', va='center', size=12)
        
        plt.xlim(-hex_size, self.cols * hex_size * 3/2)
        plt.ylim(-hex_size, self.rows * np.sqrt(3) * hex_size)
        plt.axis('off')
        plt.show()

# Example usage
hex_matrix = HexagonalMatrix(5, 5)
hex_matrix.update_value(2,2, 1)
hex_matrix.update_neighbors(2, 2, 2)
hex_matrix.display_matrix()
