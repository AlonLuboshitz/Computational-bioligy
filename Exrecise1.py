import tkinter as tk
import random
from collections import Counter
labels = []
labels_colors_before = []
labels_colors_after = []
rows = 50
columns = 50
color_dict ={"black": "white", "white": "black"}

"""
create initial board with 50-50 ratio of black and whote labels (cells).
saves chosen colors for each node in the matrix -labels_colors_before for downstream anaylsis.
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
        labels_colors_before.append(color_row)
    return labels


def get_nieghbours(i,j):
    '''function gets i,j, lokk for the  colors of the 8 niegerbours clock wise and splits it to 2 vectors:
        color_vec_column - neighbors above or below the cell
        color_voc_rows - all the other vectors
    ----- 
    returns color_vec = [(i-1,j-1).....(i,j-1)]'''
    color_vec_column = []
    
    color_voc_rows = []
    # check if the nieghbours are in the board
    if i > 0 and i < (rows-1) and j > 0 and j < (columns-1):
        color_vec_column.append(labels_colors_before[i+1][j])
        color_vec_column.append(labels_colors_before[i-1][j])
        color_voc_rows.append(labels_colors_before[i-1][j-1])
        color_voc_rows.append(labels_colors_before[i-1][j+1])
        color_voc_rows.append(labels_colors_before[i][j+1])
        color_voc_rows.append(labels_colors_before[i+1][j+1])
        color_voc_rows.append(labels_colors_before[i+1][j-1])
        color_voc_rows.append(labels_colors_before[i][j-1])
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

def change_color_by_column(current_color,nieghbours_vec):
    '''Function to change the node color by the niehbrous in the colmuns'''
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
resets the labels_colors_before matrix to the new given colors
"""
def switch_color_labels():
    for i,row in enumerate(labels):
        for j,label in enumerate(row):
            color = label.cget("bg")
            labels_colors_before[i][j] = color


"""
this function runs recursivley the function inverse_colors and apply it on the grid each x ms
"""
def recu_inverse():
    inverse_colors()
    x=10
    root.after(x, recu_inverse)

"""
prints which botton has been pressed on the grid
"""
def on_button_click(row, col):
    print(f"Button clicked at ({row}, {col})")



if __name__ == "__main__":
    # Create the main window
    root = tk.Tk()
    root.title("10x10 Board")
    create_board()
    recu_inverse()
    # for i in range(250):
    #     inverse_colors()

    root.mainloop()

