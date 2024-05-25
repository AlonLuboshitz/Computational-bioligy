import tkinter as tk
import random
from collections import Counter
labels = []
labels_colors_before = []
labels_colors_after = []
rows = 50
columns = 50
color_dict ={"black": "white", "white": "black"}
 
def create_board():
    
    for row in range(rows):
        row_labels = []
        color_row =[]
        for col in range(columns):
            # Randomly choose between "black" and "white" with the given ratio
            color = random.choices(["black", "white"], weights=[0.5, 0.5])[0]
            color_row.append(color)
            label = tk.Label(root, bg=color, borderwidth=1, relief="solid", width=8, height=2)
            label.grid(row=row, column=col)
            row_labels.append(label)
        labels.append(row_labels)
        labels_colors_before.append(color_row)
    
    return labels
def get_nieghbours(i,j):
    '''functiong ets i,j returns vector with colors of the 8 niegerbours clock wise
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
def color_count_column(column_vec,row_vec):
    counter_column = Counter(column_vec)
    counter_row = Counter(row_vec)
    black = 0.4*(counter_column["black"]/2) + 0.6*(counter_row["white"]/6)
    color = random.choices(["black", "white"], weights=[black, 1-black])[0]
    return color  
def change_color_by_column(current_color,nieghbours_vec):
    '''Function to change the node color by the niehbrous in the colmuns'''
    colors = []
    colors.append(nieghbours_vec[1])
    colors.append(nieghbours_vec[5])
    color = random.choices(colors)
    return color
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

def switch_color_labels():
    for i,row in enumerate(labels):
        for j,label in enumerate(row):
            color = label.cget("bg")
            labels_colors_before[i][j] = color


# def get_label_color(row, col):
#     if 0 <= row < len(labels) and 0 <= col < len(labels[0]):
#         return labels[row][col].cget("bg")
#     return None
def recu_inverse():
    inverse_colors()
    root.after(1, recu_inverse)
def on_button_click(row, col):
    print(f"Button clicked at ({row}, {col})")

# Create the main window
root = tk.Tk()
root.title("10x10 Board")
create_board()
#recu_inverse()
for i in range(250):
    inverse_colors()

root.mainloop()

if __name__ == "__main__":
    on_button_click(0, 0)
