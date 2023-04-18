import os
from matplotlib import pyplot as plt
import torch
import numpy as np
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import torch
from treelib import Tree

def plot_matrix(title:str, matrix, xlabel:str=None, ylabel:str=None, save: str = None):
    fig = plt.figure(figsize=(8, 6),dpi=100)
    ax = fig.add_subplot(111)
    c = ax.pcolormesh(matrix)
    ax.invert_yaxis()
    ax.set_title(f"{title}")
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    plt.colorbar(c)

    if save:
        save_dir, fname = os.path.split(save)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        plt.savefig(title)
    else:
        plt.show()
    
    plt.close()
    
def plot_hist(title:str, data, xlabel:str=None, ylabel:str=None, compare=None, save: str = None):
    fig = plt.figure(figsize=(8, 6),dpi=100)
    ax = fig.add_subplot(111)
    ax.hist(data, bins=50)
    ax.set_title(title)
    avg = np.mean(data)
    ax.axvline(avg, color="red", linestyle="dashed", linewidth=2)
    if compare:
        ax.axvline(compare, color="green", linestyle="dashed", linewidth=2)
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    if save:
        save_dir, fname = os.path.split(save)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        plt.savefig(save)
    else:
        plt.show()
    
    plt.close()
    
def plot_bar(title:str, data, xlabel:str=None, ylabel:str=None, save: str = None):
    fig = plt.figure(figsize=(8, 6),dpi=100)
    ax = fig.add_subplot(111)
    data = np.squeeze(data)
    ax.bar(range(len(data)), data)
    ax.set_title(title)
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    if save:
        save_dir, fname = os.path.split(save)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        plt.savefig(save)
    else:
        plt.show()
    
    plt.close()
    
def plotly_matrix(title: str, matrix, xlabel: str = None, ylabel: str = None, save: str = None, xtick_labels=None, ytick_labels=None):
    fig = make_subplots(rows=1, cols=1)
    heatmap = go.Heatmap(z=matrix, x=xtick_labels, y=ytick_labels, zmin=0)
    fig.add_trace(heatmap)
    fig.update_layout(title=title)
    layout = go.Layout(height=10*matrix.shape[0])
    fig.update_layout(layout)
    if xlabel:
        fig.update_xaxes(title_text=xlabel)
    if ylabel:
        fig.update_yaxes(title_text=ylabel)

    if save:
        fig.write_image(save)
    else:
        fig.show()
def plotly_hist(title:str, data, xlabel:str=None, ylabel:str=None, compare=None, save: str = None):
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=data, nbinsx=50))
    fig.update_layout(title=title)
    avg = np.mean(data)
    fig.add_shape(type="line", x0=avg, x1=avg, y0=0, y1=1, line=dict(color="red", dash="dash"))
    if compare:
        fig.add_shape(type="line", x0=compare, x1=compare, y0=0, y1=1, line=dict(color="green", dash="dash"))
    if xlabel:
        fig.update_xaxes(title_text=xlabel)
    if ylabel:
        fig.update_yaxes(title_text=ylabel)

    if save:
        fig.write_image(save)
    else:
        fig.show()

def plotly_bar(title:str, data, xlabel:str=None, ylabel:str=None, save: str = None):
    fig = go.Figure()
    fig.add_trace(go.Bar(x=list(range(len(data))), y=data))
    fig.update_layout(title=title)
    if xlabel:
        fig.update_xaxes(title_text=xlabel)
    if ylabel:
        fig.update_yaxes(title_text=ylabel)

    if save:
        fig.write_image(save)
    else:
        fig.show()

def print_struct(data):
    def build_tree(data, tree, parent=None):
        for key, value in data.items() if isinstance(data, dict) else enumerate(data):
            if isinstance(value, list):
                node = tree.create_node(tag='list', parent=parent)
                build_tree(value, tree, parent=node.identifier)
            elif isinstance(value, tuple):
                node = tree.create_node(tag='tuple', parent=parent)
                build_tree(list(value), tree, parent=node.identifier)
            elif isinstance(value, dict):
                node = tree.create_node(tag='dict', parent=parent)
                build_tree(value, tree, parent=node.identifier)
            elif isinstance(value, torch.Tensor):
                node = tree.create_node(tag=f'torch.Tensor({list(value.shape)} device={value.device})', parent=parent)
            else:
                node = tree.create_node(tag=f'{type(value).__name__}', parent=parent)
            if isinstance(data, dict):
                node.tag = f'"{key}": {node.tag}'
        return tree

    tree = Tree()
    tree.create_node(tag='root', identifier=0)
    build_tree(data, tree, parent=0)
    tree.show()