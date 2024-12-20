import cv2
import os
import numpy as np
import tkinter as tk
import matplotlib.pyplot as plt
import helper_functions as hf
from area_mapping import mask_deployment, get_tree_mask, get_water_mask
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from tkinter import ttk

# Function to update the plot based on the image
def update_plot(loading=False):
    overlay = img.copy()
    alpha: float = alpha_var.get()

    active_masks = []

    if coast_var.get():
        active_masks.append((coast_mask, (174, 235, 52), alpha))

    if inland_var.get():
        active_masks.append((inland_mask, (245, 130, 37), alpha))

    if forest_edge_var.get():
        active_masks.append((forest_edge_mask, (81, 153, 14), alpha))

    if tree_var.get():
        active_masks.append((tree_mask, (66, 191, 50), alpha))

    if water_var.get():
        active_masks.append((water_mask, (58, 77, 222), alpha))

    overlay = hf.overlay_from_masks(overlay, *active_masks)

    # Update matplotlib plot
    ax.clear()

    # Show "Loading..." text when loading flag is set
    ax.text(0.5, 0.5, 'Loading...' if loading else "", color='black', fontsize=18, ha='center', va='center', transform=ax.transAxes)

    ax.imshow(overlay)

    # Display paths if path layer is enabled
    if path_var.get():
        for path_points in paths_points:
            if path_points is not None:
                for i, point in enumerate(path_points):
                    if i > 0:
                        x1, y1 = path_points[i - 1]
                        x2, y2 = point
                        line = plt.Line2D(
                            [x1, x2], [y1, y2],
                            linewidth=3,
                            color=(0.7, 0.7, 0.7) if point not in bridge_points else (0.8, 0.6, 0.4)
                        )
                        ax.add_line(line)  # Add the path line

    # Display buildings if building layer is enabled
    if building_var.get():
        for building in buildings:
            x, y, w, h = building["rect"]
            rect = plt.Rectangle(
                (x, y), w, h,
                linewidth=1, edgecolor="white", facecolor="none"
            )
            ax.add_patch(rect)  # Draw the building rectangle
            ax.text(
                x + w / 2, y - 5,
                building["nametag"],
                color="white", fontsize=6, ha="center"
            )

    canvas.draw()

# Function to load a new image based on the selection in the dropdown
def load_image(event=None):
    update_plot(loading=True)  # Show "Loading..." text before loading masks

    global img, coast_mask, inland_mask, forest_edge_mask, tree_mask, water_mask, buildings, paths_points, bridge_points
    
    # Delay loading and calculating masks to ensure the "Loading..." text is shown
    def update_masks():
        global img, coast_mask, inland_mask, forest_edge_mask, tree_mask, water_mask, buildings, paths_points, bridge_points
        
        # Load the selected image
        img_path = os.path.join("./mocking_examples", image_selection.get())
        img = cv2.imread(img_path)
        
        # Calculate masks for the new image
        coast_mask, inland_mask, forest_edge_mask, tree_mask, water_mask, buildings, paths_points, bridge_points = mask_deployment(
            get_tree_mask(img_path), get_water_mask(img_path)
        )
        
        # Update plot with new masks
        update_plot(loading=False)  # Remove "Loading..." text and show new masks

    # Delay mask loading using `after` to ensure the text is shown
    root.after(100, update_masks)

# Create Tkinter main window
root = tk.Tk()
root.title("Cartography Dashboard")

# Sidebar
sidebar = tk.Frame(root, width=250, bg="#1f1f1f")
sidebar.pack(side=tk.LEFT, fill=tk.Y)

# Variables for masks
coast_var = tk.BooleanVar(value=True)
inland_var = tk.BooleanVar(value=True)
forest_edge_var = tk.BooleanVar(value=True)
tree_var = tk.BooleanVar(value=True)
water_var = tk.BooleanVar(value=True)
building_var = tk.BooleanVar(value=True)
path_var = tk.BooleanVar(value=True)

style = ttk.Style()

dark_bg = "#2e2e2e"
light_bg = "#3a3a3a"
text_color = "#828282"
highlight_color = "#5a5a5a"

# Checkbox Style
style.configure(
    "Dark.TCheckbutton",
    background=dark_bg,
    foreground=text_color,
    font=("Arial", 10),
    indicatorcolor=light_bg,
    indicatormargin=5,
    indicatordiameter=15,
)

# Checkbox Hover Style
style.map(
    "Dark.TCheckbutton",
    background=[("active", dark_bg)],
    indicatorcolor=[("active", light_bg), ("!active", dark_bg)],
)

# Label Style
style.configure(
    "Dark.TLabel",
    background=dark_bg,
    foreground=text_color,
    font=("Arial", 12, "bold")
)

# Dropdown menu for selecting an image
image_files = [f for f in os.listdir("./mocking_examples") if f.endswith(".png")]
image_selection = ttk.Combobox(sidebar, values=image_files, style="TCombobox")
image_selection.pack(padx=10, pady=(10, 5))
image_selection.bind("<<ComboboxSelected>>", load_image)

# Splitter
canvas = tk.Canvas(sidebar, width=150, height=1, bg="gray", bd=0, highlightthickness=0)
canvas.pack(padx=10, pady=10)

# Labels and checkbuttons for masks
ttk.Label(sidebar, text="Toggle Layers", style="Dark.TLabel").pack(anchor="w", padx=10, pady=5)
ttk.Checkbutton(sidebar, text="Coast Mask", variable=coast_var, style="Dark.TCheckbutton", command=update_plot).pack(anchor="w", padx=10, pady=5)
ttk.Checkbutton(sidebar, text="Inland Mask", variable=inland_var, style="Dark.TCheckbutton", command=update_plot).pack(anchor="w", padx=10, pady=5)
ttk.Checkbutton(sidebar, text="Forest Edge Mask", variable=forest_edge_var, style="Dark.TCheckbutton", command=update_plot).pack(anchor="w", padx=10, pady=5)
ttk.Checkbutton(sidebar, text="Tree Mask", variable=tree_var, style="Dark.TCheckbutton", command=update_plot).pack(anchor="w", padx=10, pady=5)
ttk.Checkbutton(sidebar, text="Water Mask", variable=water_var, style="Dark.TCheckbutton", command=update_plot).pack(anchor="w", padx=10, pady=5)
ttk.Checkbutton(sidebar, text="Buildings", variable=building_var, style="Dark.TCheckbutton", command=update_plot).pack(anchor="w", padx=10, pady=5)
ttk.Checkbutton(sidebar, text="Paths", variable=path_var, style="Dark.TCheckbutton", command=update_plot).pack(anchor="w", padx=10, pady=5)

# Splitter
canvas = tk.Canvas(sidebar, width=150, height=1, bg="gray", bd=0, highlightthickness=0)
canvas.pack(padx=10, pady=10)

# Title Label
ttk.Label(sidebar, text="Adjust Alpha", style="Dark.TLabel").pack(anchor="w", padx=10, pady=5)

# Transparency Adjustment Slider
alpha_var = tk.DoubleVar(value=0.35)  # Default alpha value
alpha_slider = ttk.Scale(
    sidebar, from_=0.0, to=1.0, orient="horizontal", variable=alpha_var,
    style="Horizontal.TScale", command=lambda val: update_plot()
)
alpha_slider.pack(anchor="w", padx=10, pady=5)

# Load and display the default image
img_path = os.path.join("./mocking_examples", image_files[0])  # Select first image
img = cv2.imread(img_path)

# Calculate masks for the first image (predefine globals)
coast_mask, inland_mask, forest_edge_mask, tree_mask, water_mask, buildings, paths_points, bridge_points = mask_deployment(
    get_tree_mask(img_path), get_water_mask(img_path)
)

# Create matplotlib plot
fig = Figure(figsize=(8, 6))
ax = fig.add_subplot(111)

fig.patch.set_facecolor('#2e2e2e')

axes_colors = '#7d7d7d'
ax.spines['bottom'].set_color(axes_colors)
ax.spines['left'].set_color(axes_colors)
ax.spines['top'].set_color(axes_colors)
ax.spines['right'].set_color(axes_colors)
ax.tick_params(colors=axes_colors)
ax.xaxis.label.set_color(axes_colors)
ax.yaxis.label.set_color(axes_colors)
ax.title.set_color(axes_colors)

canvas = FigureCanvasTkAgg(fig, master=root)
canvas_widget = canvas.get_tk_widget()
canvas_widget.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

# Initialize plot
update_plot()

root.mainloop()
