import tkinter as tk

def on_button_click(row, col):
    print(f"Button clicked at ({row}, {col})")

# Create the main window
root = tk.Tk()
root.title("10x10 Board")

# Create a 10x10 grid of buttons
for row in range(10):
    for col in range(10):
        button = tk.Button(root, text=f"{row},{col}", command=lambda r=row, c=col: on_button_click(r, c))
        button.grid(row=row, column=col, padx=5, pady=5)

# Start the GUI event loop
root.mainloop()
