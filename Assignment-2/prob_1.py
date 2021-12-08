import numpy as np
import pandas as pd
def minkowski(power, points):

    return np.power(np.sum(np.power(np.abs(np.diff(points, axis=0)), power)), (1/power))

def minkowski_vector(power, A,B):
    return np.power(np.sum(np.power(np.abs(A-B),power),axis=1),(1/power))



def knn(data,point,k=3):
    return data[np.argsort(minkowski_vector(2,data[:,:-1],point))[:k]][:,-1].mean()



if __name__ == "__main__":
    classes = ['A', 'B']

    dataset_1a = pd.read_csv('dataSet1a.csv', names=['x', 'y'])
    dataset_1b = pd.read_csv('dataSet1b.csv', names=['x', 'y'])

    x_mean_1 = dataset_1a['x'].mean()
    y_mean_1 = dataset_1a['y'].mean()

    x_mean_2 = dataset_1b['x'].mean()
    y_mean_2 = dataset_1b['y'].mean()

    point = np.array([3, 3], dtype=np.float64)
    


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
    
    dataset_1a['class'] = 0
    dataset_1b['class'] = 1
    datasets = pd.concat([dataset_1a,dataset_1b],axis=0,ignore_index=True).to_numpy()

    k=3
    confidence = knn(datasets,point,k=k)
    print(
        f"KNN (k={k}) Classification: Class {classes[round(confidence)]} confidence: {confidence:.2f}")

    k=1
    confidence = knn(datasets,point,k=k)
    print(
        f"KNN (k={k}) Classification: Class {classes[round(confidence)]} confidence: {confidence:.2f}")
    k=5
    confidence = knn(datasets,point,k=k)
    print(
        f"KNN (k={k}) Classification: Class {classes[round(confidence)]} confidence: {confidence:.2f}")
    k=7
    confidence = knn(datasets,point,k=k)
    print(
        f"KNN (k={k}) Classification: Class {classes[round(confidence)]} confidence: {confidence:.2f}")
    k=11
    confidence = knn(datasets,point,k=k)
    print(
        f"KNN (k={k}) Classification: Class {classes[round(confidence)]} confidence: {confidence:.2f}")
    k=13
    confidence = knn(datasets,point,k=k)
    print(
        f"KNN (k={k}) Classification: Class {classes[round(confidence)]} confidence: {confidence:.2f}")
    k=17
    confidence = knn(datasets,point,k=k)
    print(
        f"KNN (k={k}) Classification: Class {classes[round(confidence)]} confidence: {confidence:.2f}")
    k=21
    confidence = knn(datasets,point,k=k)
    print(
        f"KNN (k={k}) Classification: Class {classes[round(confidence)]} confidence: {confidence:.2f}")
    k=30
    confidence = knn(datasets,point,k=k)
    print(
        f"KNN (k={k}) Classification: Class {classes[round(confidence)]} confidence: {confidence:.2f}")
    k=100
    confidence = knn(datasets,point,k=k)
    print(
        f"KNN (k={k}) Classification: Class {classes[round(confidence)]} confidence: {confidence:.2f}")
    k=120
    confidence = knn(datasets,point,k=k)
    print(
        f"KNN (k={k}) Classification: Class {classes[round(confidence)]} confidence: {confidence:.2f}")
    k=150
    confidence = knn(datasets,point,k=k)
    print(
        f"KNN (k={k}) Classification: Class {classes[round(confidence)]} confidence: {confidence:.2f}")
    


