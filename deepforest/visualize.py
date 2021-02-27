#Visualize module for plotting and handling predictions
import os
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from skimage import io
import numpy as np

def format_predictions(prediction):
    """Format a retinanet prediction into a pandas dataframe for a single image"""
    df = pd.DataFrame(prediction["boxes"].cpu().detach().numpy(),columns=["xmin","ymin","xmax","ymax"])
    df["label"] = prediction["labels"].cpu().detach().numpy()
    df["scores"] = prediction["scores"].cpu().detach().numpy()
    
    return df

def plot_prediction_dataframe(df, ground_truth, root_dir, savedir):
    """For each row in dataframe, call plot predictions"""
    for name, group in df.groupby("image_path"):
        image = io.imread("{}/{}".format(root_dir,name))
        plot, ax = plot_predictions(image, group)
        annotations = ground_truth[ground_truth.image_path==name]
        plot = add_annotations(plot, ax, annotations)
        plot.savefig("{}/{}.png".format(savedir,os.path.splitext(name)[0]))        
        
def plot_predictions(image, df):
    fig, ax = plt.subplots()
    ax.imshow(image)
    for index, row in df.iterrows():
        xmin = row["xmin"]
        ymin = row["ymin"]
        width = row["xmax"] - xmin
        height = row["ymax"] - ymin
        color = label_to_color(row["label"])
        rect = create_box(xmin=xmin,ymin=ymin, height=height, width=width,color=color)
        ax.add_patch(rect)
    #no axis show up
    plt.axis('off')

    return fig, ax
        
def create_box(xmin, ymin, height, width, color="cyan",linewidth=1):
    rect = patches.Rectangle((xmin,ymin),
                     height,
                     width,
                     linewidth=linewidth,
                     edgecolor=color,
                     fill = False)
    return rect

def add_annotations(plot, ax, annotations):
    """Add annotations to an already created visuale.plot_predictions
    Args:
        plot: matplotlib figure object
        ax: maplotlib axes object
        annotations: pandas dataframe of bounding box annotations
    Returns:
        plot: matplotlib figure object
    """
    for index, row in annotations.iterrows():
        xmin = row["xmin"]
        ymin = row["ymin"]
        width = row["xmax"] - xmin
        height = row["ymax"] - ymin
        rect = create_box(xmin=xmin,ymin=ymin, height=height, width=width, color="orange")
        ax.add_patch(rect) 
    
    return plot

def label_to_color(label):
    color_dict = {}
    colors = [list((matplotlib.colors.hsv_to_rgb([x, 1.0, 1.0]) * 255).astype(int)) for x in np.arange(0, 1, 1.0 / 80)]
    for index, color in enumerate(colors):
        color_dict[index] = color
    
    #hand pick the first few colors
    color[0] = "cyan"
    color[1] = "tomato"
    color[2] = "blue"
    color[3] = "limegreen"
    color[4] = "orchid"
    color[5] = "crimson"
    color[6] = "peru"
    color[7] = "dodgerblue"
    color[8] = "gold"
    color[9] = "blueviolet"
    
    return color_dict[label]