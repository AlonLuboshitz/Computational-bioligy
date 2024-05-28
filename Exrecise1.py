import tkinter as tk
import random
from collections import Counter
import sys

#global variables
labels = []
labels_colors = []
rows = 50
columns = 50
color_dict ={"black": "white", "white": "black"}
scores =[]

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
            label = tk.Label(root, bg=color, borderwidth=1, relief="solid", width=8, height=2)
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
    if i > 0 and i < (rows-1) and j > 0 and j < (columns-1):
        color_vec_column.append(labels_colors[i+1][j])
        color_vec_column.append(labels_colors[i-1][j])
        color_voc_rows.append(labels_colors[i-1][j-1])
        color_voc_rows.append(labels_colors[i-1][j+1])
        color_voc_rows.append(labels_colors[i][j+1])
        color_voc_rows.append(labels_colors[i+1][j+1])
        color_voc_rows.append(labels_colors[i+1][j-1])
        color_voc_rows.append(labels_colors[i][j-1])
        return color_vec_column,color_voc_rows
    return 0,0

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
            col_vec,row_vec = get_nieghbours(i,j)
            if col_vec and row_vec:
                inverse_color = color_count_column(col_vec,row_vec)
                label.config(bg=inverse_color)
    root.update()
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
        count = 0
        for i in range(rows):
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
        col_scores.append(temp_max)
    return (sum(col_scores) / columns)

"""
    calculate the score for the rows:
    averge of the number of "switching colors" in each row
    for exmp: *white, black, white*, white, *black* --> 4
"""
def row_score():
    row_scores= []
    for i in range(rows):
        count = 0
        temp_color = labels_colors[i][0]
        for j in range(1, columns):
            color = labels_colors[i][j]
            if color != temp_color:
                count+=1
                temp_color = color

        row_scores.append(count)
    return (sum(row_scores) / rows)

"""
calculate the board score according to the col_score and row_score, and appends it to the score array
"""
def total_score():
    score = (0.4 * col_score()) + (0.6 * row_score())
    scores.append(score)
"""
this function runs recursivley the function inverse_colors and apply it on the grid each x ms
for each run it calculates the board score
"""
def recu_inverse():
    
    inverse_colors()
    x=10
    total_score()
    print(scores)
    root.after(x, recu_inverse)


"""
prints which botton has been pressed on the grid
"""
def on_button_click(row, col):
    print(f"Button clicked at ({row}, {col})")



if __name__ == "__main__":
    output_file = sys.argv[1]
    # Create the main window
    root = tk.Tk()
    root.title("cell automata")

    #create boad and run the rules
    create_board()
    recu_inverse()

    # show board until closed
    root.mainloop()

    # print scores to file
    with open(output_file, 'w') as file:
        for run, score in enumerate(scores):
            file.write(f"{run}: {score}\n")  # Write each element on a new line