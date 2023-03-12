from matplotlib import pyplot as plt
import torch
import numpy as np

def plot_layer_attn(title:str, attention_weights:torch.Tensor):
    """
    attention_weights: [bsz, num_heads, num_tokens, num_tokens]
    """
    for i in range(attention_weights.shape[0]):
        for head in range(attention_weights.shape[1]):
            plot_matrix(f"{title}_layer_{i}_head_{head}", attention_weights[i,head])
            
def plot_matrix(title:str, matrix, xlabel:str=None, ylabel:str=None):
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
    plt.savefig(f"{title}.png")
    plt.close()
    
def plot_hist(title:str, data, compare=None):
    fig = plt.figure(figsize=(8, 6),dpi=100)
    ax = fig.add_subplot(111)
    ax.hist(data, bins=50)
    ax.set_title(f"{title}")
    avg = np.mean(data)
    ax.axvline(avg, color="red", linestyle="dashed", linewidth=2)
    if compare:
        ax.axvline(compare, color="green", linestyle="dashed", linewidth=2)
    plt.savefig(f"{title}.png")
    plt.close()
    
def plot_bar(title:str, data):
    fig = plt.figure(figsize=(8, 6),dpi=100)
    ax = fig.add_subplot(111)
    data = np.squeeze(data)
    ax.bar(range(len(data)), data)
    ax.set_title(f"{title}")
    plt.savefig(title)
    plt.close()