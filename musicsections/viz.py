import json
import matplotlib.pyplot as plt
from mir_eval import display
import numpy as np
import os


def plot_levels(inters, labels, figsize, rotation):
    """Plots the given hierarchy."""
    N = len(inters)
    fig, axs = plt.subplots(N, figsize=figsize)
    for level in range(N):
        display.segments(np.asarray(inters[level]), labels[level], ax=axs[level])
        axs[level].set_yticks([0.5])
        axs[level].set_yticklabels([N - level])
        axs[level].set_xticks([])
    axs[0].xaxis.tick_top()
    fig.subplots_adjust(top=0.8)  # Otherwise savefig cuts the top
    
    return fig, axs

def load_segmentation(json_file):
    with open(json_file, "r") as f:
        seg = json.load(f)
    return seg

def plot_segmentation(seg, figsize=(13, 3), rotation=45):
    inters = []
    labels = []
    for level in seg[::-1]:
        inters.append(level[0])
        labels.append(level[1])
    fig, axs = plot_levels(inters, labels, figsize, rotation)
    fig.text(0.08, 0.47, 'Segmentation Levels', va='center', rotation='vertical')

def plot_segmentation_json(json_path):
    seg = load_segmentation(json_path)
    plot_segmentation(seg, figsize=(13, 3))
