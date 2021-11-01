import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import norm

def parametric(X):
    mean = np.mean(X)
    std = np.std(X,ddof=1)
    D = np.linspace(np.min(X), np.max(X))
    fig,ax = plt.subplots()
    ax.plot(D,norm.pdf(D,mean,std), label=f'$( \mu \\equiv {mean} , \sigma \\equiv {std} )$')
    ax.hist(X,edgecolor='black',alpha=.5,density=True)
    ax.set_title('Parametric Density Function')
    ax.set_xlabel('Value')
    ax.set_ylabel('Density')
    ax.legend()

    mode = mean
    print(f"Dataset 2 parametric modes: {mode}")

def non_parametric(X, binwidth):
    fig,ax = plt.subplots()
    (n,bins, patches) = ax.hist(X,bins=np.arange(np.min(X),np.max(X)+binwidth,binwidth),edgecolor='black')
    ax.set_title('Non Parametric Density Function')
    ax.set_xlabel('Value')
    ax.set_ylabel('Density')

    modes = bins[np.argwhere(n == np.amax(n))].flatten()
    print(f"Dataset 2 non-parametric modes: {modes}")
    
if __name__ == "__main__":
    dataset2 = pd.read_csv('dataSet2.csv', names=['x'])
    parametric(dataset2['x'])
    non_parametric(dataset2['x'],binwidth=0.40)
    plt.show()