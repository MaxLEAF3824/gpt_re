import os
from matplotlib import pyplot as plt
import torch
import numpy as np
import plotly.graph_objs as go
from plotly.subplots import make_subplots


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
    
def plotly_matrix(title:str, matrix, xlabel:str=None, ylabel:str=None, save: str = None):
    fig = make_subplots(rows=1, cols=1)
    heatmap = go.Heatmap(z=matrix, zmin=0)
    fig.add_trace(heatmap)
    fig.update_layout(title=title)
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
