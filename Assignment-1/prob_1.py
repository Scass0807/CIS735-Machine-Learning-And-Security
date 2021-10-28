import numpy as np
from numpy.lib.function_base import diff
import pandas as pd


def minkowski(power,points):
    
    return np.power(np.sum(np.power(np.abs(np.diff(points,axis=0)),power)),(1/power))


if __name__ ==  "__main__":
    classes = ['A', 'B']

    dataset_1a = pd.read_csv('dataSet1a.csv',names=['x','y'])
    dataset_1b = pd.read_csv('dataSet1b.csv',names=['x','y'])
    
    x_mean_1 = dataset_1a['x'].mean()
    y_mean_1 = dataset_1a['y'].mean()

    x_mean_2 = dataset_1b['x'].mean()
    y_mean_2 = dataset_1b['y'].mean()

    point = np.array([3,3],dtype=np.float64)

    manhattan_1 = minkowski(1,np.array([[x_mean_1,y_mean_1],point],dtype=np.float64))
    manhattan_2 = minkowski(1,np.array([[x_mean_2,y_mean_2],point],dtype=np.float64))

    print(f'Manhattan Distance between the point {point} and Class A: {manhattan_1}')
    print(f'Manhattan Distance between the point {point} and Class B: {manhattan_2}')
    print(f"Manhattan Classification: Class {classes[np.argmin(np.array([manhattan_1,manhattan_2]))]}") 

    euclidean_1 = minkowski(2,np.array([[x_mean_1,y_mean_1],point],dtype=np.float64))
    euclidean_2 = minkowski(2,np.array([[x_mean_2,y_mean_2],point],dtype=np.float64))

    print(f'Euclidean Distance between the point {point} and Class A: {euclidean_1}')
    print(f'Euclidean Distance between the point {point} and Class B: {euclidean_2}')
    print(f"Euclidean Classification: Class {classes[np.argmin(np.array([euclidean_1,euclidean_2]))]}") 