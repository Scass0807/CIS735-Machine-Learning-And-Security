import numpy as np
import pandas as pd


def minkowski(power, points):

    return np.power(np.sum(np.power(np.abs(np.diff(points, axis=0)), power)), (1/power))


def mahalanobis(x, dataset):
    x_minus_mean = x-dataset.values.mean(axis=0)
    covariance = np.cov(dataset.values.T)
    inverse_covariance = np.linalg.inv(covariance)
    return np.diag(np.sqrt(np.dot(np.dot(x_minus_mean, inverse_covariance), x_minus_mean.T)))


if __name__ == "__main__":
    classes = ['A', 'B']

    dataset_1a = pd.read_csv('dataSet1a.csv', names=['x', 'y'])
    dataset_1b = pd.read_csv('dataSet1b.csv', names=['x', 'y'])

    x_mean_1 = dataset_1a['x'].mean()
    y_mean_1 = dataset_1a['y'].mean()

    x_mean_2 = dataset_1b['x'].mean()
    y_mean_2 = dataset_1b['y'].mean()

    point = np.array([3, 3], dtype=np.float64)
    print("\n1A")
    manhattan_1 = minkowski(1, np.array(
        [[x_mean_1, y_mean_1], point], dtype=np.float64))
    manhattan_2 = minkowski(1, np.array(
        [[x_mean_2, y_mean_2], point], dtype=np.float64))

    print(
        f'Manhattan Distance between the point {point} and Class A: {manhattan_1}')
    print(
        f'Manhattan Distance between the point {point} and Class B: {manhattan_2}')
    print(
        f"Manhattan Classification: Class {classes[np.argmin(np.array([manhattan_1,manhattan_2]))]}")

    euclidean_1 = minkowski(2, np.array(
        [[x_mean_1, y_mean_1], point], dtype=np.float64))
    euclidean_2 = minkowski(2, np.array(
        [[x_mean_2, y_mean_2], point], dtype=np.float64))

    print(
        f'Euclidean Distance between the point {point} and Class A: {euclidean_1}')
    print(
        f'Euclidean Distance between the point {point} and Class B: {euclidean_2}')
    print(
        f"Euclidean Classification: Class {classes[np.argmin(np.array([euclidean_1,euclidean_2]))]}")

    mahalanobis_1 = mahalanobis(
        np.array([point], dtype=np.float64), dataset_1a[['x', 'y']]).item()
    mahalanobis_2 = mahalanobis(
        np.array([point], dtype=np.float64), dataset_1b[['x', 'y']]).item()

    print(
        f'Mahalanobis Distance between the point {point} and Class A: {mahalanobis_1}')
    print(
        f'Mahalanobis Distance between the point {point} and Class B: {mahalanobis_2}')
    print(
        f"Mahalanobis Classification: Class {classes[np.argmin(np.array([mahalanobis_1,mahalanobis_2]))]}")

    p_values = [0.01, 0.05, 0.1, 0.25, 0.5, 1,
                1.5, 2, 3, 5, 10, 25, 50, 100, 1000]

    print("\n1B")
    for p in p_values:
        minkowski_1 = minkowski(p, np.array(
        [[x_mean_1, y_mean_1], point], dtype=np.float64))
        minkowski_2 = minkowski(p, np.array(
        [[x_mean_2, y_mean_2], point], dtype=np.float64))
        print(f'Minkowski p={p} Distance between the point {point} and Class A: {minkowski_1}')
        print(f'Minkowski p={p} Distance between the point {point} and Class B: {minkowski_2}')
        print(f"Minkowski Classification: Class {classes[np.argmin(np.array([minkowski_1,minkowski_2]))]}") 
