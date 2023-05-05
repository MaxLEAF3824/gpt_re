import os
from matplotlib import pyplot as plt
import torch
import numpy as np
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import torch
from treelib import Tree

 
def plotly_matrix(matrix, title='',  xlabel: str = None, ylabel: str = None, save: str = None, xtick_labels=None, ytick_labels=None):
    fig = make_subplots(rows=1, cols=1)
    heatmap = go.Heatmap(z=matrix, x=xtick_labels, y=ytick_labels, zmin=0)
    fig.add_trace(heatmap)
    if title:
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

def plotly_hist(data, title='', xlabel:str=None, ylabel:str=None, compare=None, save: str = None, bins=50):
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=data, nbinsx=bins))
    avg = np.mean(data)
    fig.add_shape(type="line", x0=avg, x1=avg, y0=0, y1=1, line=dict(color="red", dash="dash"))
    if title:
        fig.update_layout(title=title)
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

def plotly_bar(data, title='',  xlabel:str=None, ylabel:str=None, save: str = None):
    fig = go.Figure()
    fig.add_trace(go.Bar(x=list(range(len(data))), y=data))
    if title:
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