import tkinter as tk
import random
from collections import Counter
import numpy as np
from tkinter import ttk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
#global variables
all_scores = []
labels = []
labels_colors = []
rows = 80
columns = 80
color_dict ={"black": "white", "white": "black"}
scores =[]
iteration_counter = 0
max_iterations = 250

"""
create initial board with 50-50 ratio of black and whote labels (cells).
saves chosen colors for each node in the matrix -labels_colors for downstream anaylsis.
------
returns labels matrix
"""
def create_board():
    for row in range(rows):
        row_labels = []
        color_row =[]
        for col in range(columns):
            # Randomly choose between "black" and "white" with the given ratio, and save it
            color = random.choices(["black", "white"], weights=[0.5, 0.5])[0]
            color_row.append(color)

            # create label and save it
            label = tk.Label(board_frame, bg=color, borderwidth=1, relief="solid", width=1, height=1)
            label.grid(row=row, column=col)
            row_labels.append(label)

        labels.append(row_labels)
        labels_colors.append(color_row)
    return labels

"""
function gets i,j, lokk for the  colors of the 8 niegerbours clock wise and splits it to 2 vectors:
    color_vec_column - neighbors above or below the cell
    color_voc_rows - all the other vectors
----- 
returns color_vec = [(i-1,j-1).....(i,j-1)]
"""
def get_nieghbours(i,j):

    color_vec_column = []
    
    color_voc_rows = []
    # check if the nieghbours are in the board


    color_vec_column.append(labels_colors[(i+1)%rows][j])
    color_vec_column.append(labels_colors[(i-1)%rows][j])
    color_voc_rows.append(labels_colors[(i-1)%rows][(j-1)%columns])
    color_voc_rows.append(labels_colors[(i-1)%rows][(j+1)%columns])
    color_voc_rows.append(labels_colors[i][(j+1)%columns])
    color_voc_rows.append(labels_colors[(i+1)%rows][(j+1)%columns])
    color_voc_rows.append(labels_colors[(i+1)%rows][(j-1)%columns])
    color_voc_rows.append(labels_colors[i][(j-1)%columns])
    return color_vec_column,color_voc_rows

"""
this function counts he number of neighbirs of each 'kind' and decieds which color will the cell be accrding to-
    1 -consider each black cell as 1, and each white cell as 0
    2 - count number of *black* cells of the column vec (out of 2) and compute erlative ratio
    3 - count number of *white* cells of the rows vec (out of 6) and compute erlative ratio
    4 - set black ratio as the sum of 0.4 * (2) +  0.6 * (3), (if the non columns cells are white, the cel wants
        to be the opposite- black) , ehile giving the column a 'weight' of 0.4 and others a 'weight' of 0.6
        in the decision making. 
    5 - if black ration is bigger than 0.5 - choose black
        if black ration is smaller than 0.5 - choose black
        if black ration equals to 0.5 - choose randomly between black and white
----
return the chosen color
"""

def color_count_column(column_vec,row_vec):
    # init counters
    counter_column = Counter(column_vec)
    counter_row = Counter(row_vec)
    # get ratio for black
    black_ratio = 0.4*(counter_column["black"]/2) + 0.6*(counter_row["white"]/6)
    # randomley choose with the given biased ratio
    if black_ratio > 0.5:
        color="black"
    elif black_ratio <0.5:
        color="white"
    else:
        color = random.choices(["black", "white"], weights=[0.5, 0.5])[0]
    return color 
"""
Function to change the node color by the niehbrous in the colmuns
"""
def change_color_by_column(nieghbours_vec):

    colors = []
    colors.append(nieghbours_vec[1])
    colors.append(nieghbours_vec[5])
    color = random.choices(colors)
    return color

"""
this function iterates thourgh the labels (cells) and change its color according to the decision func.
"""
def inverse_colors():
    for i,row in enumerate(labels):
    
        color_row = []
        for j,label in enumerate(row):
            color = label.cget("bg")
            color = label
            col_vec,row_vec = get_nieghbours(i,j)
            if col_vec and row_vec:
                inverse_color = color_count_column(col_vec,row_vec)
                label.config(bg=inverse_color)
                
    board_frame.update()
    switch_color_labels()

"""
resets the labels_colors matrix to the new given colors
"""
def switch_color_labels():
    for i,row in enumerate(labels):
    
        for j,label in enumerate(row):
            color = label.cget("bg")
            labels_colors[i][j] = color
"""
    calculate the score for the columns:
    averge of the maximum continous colors of each column
"""
def col_score():
    
    col_scores= []
    for j in range(columns):
        temp_color = labels_colors[0][j]
        temp_max = 0
        count = 1
        for i in range(1,rows):
            color = labels_colors[i][j]
            # if colores identical 
            if color == temp_color:
                count+=1
            # else- different colors, reset the count
            else:
                if (temp_max < count):
                    temp_max = count
                temp_color = color
                count = 1
        if (temp_max < count):
            temp_max = count
        col_scores.append(temp_max)
    avg = sum(col_scores) / (columns*columns)
    return avg

"""
    calculate the score for the rows:
    averge of the number of "switching colors" in each row
    for exmp: *white, black, white*, white, *black* --> 4
"""
def row_score():
    row_scores= []
    for i in range(rows):
        count = 1
        temp_color = labels_colors[i][0]
        for j in range(1, columns):
            color = labels_colors[i][j]
            if color != temp_color:
                count+=1
                temp_color = color

        row_scores.append(count)
    avg = sum(row_scores) / (rows*rows)
    return avg

"""
calculate the board score according to the col_score and row_score, and appends it to the score array
"""
def total_score():
    score = ((0.4 * col_score()) + (0.6 * row_score()))*100
    scores.append(score)
"""
this function runs recursivley the function inverse_colors and apply it on the grid each x ms
for each run it calculates the board score
"""
def recu_inverse():

    inverse_colors()
    total_score()
    update_plot()
    root.after(2, recu_inverse)  # Update every 2 ms


def update_plot():
    ax.clear()
    ax.plot(scores, marker='o')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Score -  %')
    ax.set_title('Score over Time')
    canvas.draw()

def plot_performance(y_values_avg, stds):
    fig ,ax = plt.subplots()
    ax.plot(range(len(y_values_avg)), y_values_avg)
    ax.fill_between(range(len(y_values_avg)), y_values_avg - stds, y_values_avg + stds, color='blue', alpha=0.3)
    ax.plot(range(len(y_values_avg)), y_values_avg - stds, linestyle='dotted', color='blue')
    ax.plot(range(len(y_values_avg)), y_values_avg + stds, linestyle='dotted', color='blue')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Score %')
    ax.set_title('Averaged score over Runs')
    ax.legend(['Average Score', 'Standard Deviation'])
    plt.show()
    plt.savefig('performance.png')
   
def extract_scores(file):
    scores = np.genfromtxt(file)
    avg = scores.mean(axis=0)
    std = scores.std(axis=0)
    return avg,std
def main_loop():
    global iteration_counter, max_iterations
    inverse_colors()
    total_score()
    update_plot()
    if iteration_counter < max_iterations - 1:
        root.after(2, main_loop)
    
    
    
  

    all_scores.append(scores)
if __name__ == "__main__":
    root = tk.Tk()
    root.title("cell automata")
        # Create a frame for the board
    board_frame = ttk.Frame(root)
    board_frame.grid(row=0, column=0, padx=10, pady=10)
    create_board()

    # Create a frame for the score plot
    plot_frame = ttk.Frame(root)
    plot_frame.grid(row=0, column=1, padx=10, pady=10)

    # # Set up the matplotlib figure and axis
    fig, ax = plt.subplots()
    canvas = FigureCanvasTkAgg(fig, master=plot_frame)
    canvas.get_tk_widget().pack()
    
    recu_inverse()  # Start the loop with a delay of 2 milliseconds

    root.mainloop()
    
    

