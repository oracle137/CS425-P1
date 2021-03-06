import numpy as np
import pandas as pd


def normal_eqn(X, Y):
  theta = np.dot(np.linalg.inv(np.dot(X.T, X)),np.dot(X.T, Y))
  return theta


def standarize(A):
    A = (A - np.mean(A, axis=0)) / np.std(A, axis=0)
    return A


def linear_regression():
    data = np.genfromtxt('auto-mpg.data.csv',usecols=(0,1,2,3,4,5,6),skip_header=True,delimiter=',')
    # print(data)
    # frame = pd.DataFrame(data)
    # print(frame.describe())
    i = 0
    while i < len(data): # remove nan rows.
        if np.isnan(data[i][3]):
            data = np.delete(data,i,0)
        i = i +1

    np.random.seed(100)
    np.random.shuffle(data)  # shuffle

    # data = standarize(data)

    X = data[:, [1,2,3,4,5,6]]#other variables
    Y = data[:,0] #MPG

    T_X = X[:int(len(data)*0.5)]
    T_Y = Y[:int(len(data)*0.5)]

    V_X = X[int(len(data)*0.5):int(len(data)*0.75)]
    V_Y = Y[int(len(data)*0.5):int(len(data)*0.75)]

    Test_X = X[int(len(data)*0.75):]
    Test_Y = Y[int(len(data)*0.75):]


    theta_T = normal_eqn(T_X, T_Y) #Training Data
    E_data = np.dot(Test_X,theta_T) #verification on Test Data
    E_data2 = np.dot(V_X, theta_T)

    error_list=[]
    sum = 0
    for i in range(0,len(E_data)):
        sum = sum + abs(Test_Y[i] - E_data[i])
        error_list.append(abs(Test_Y[i] - E_data[i]))
    avg_error = sum / len(E_data)
    print("Average Error on Test Data: ", avg_error)

    sum = 0
    for b in range(0, len(E_data2)):
        sum = sum + abs(V_Y[b] - E_data2[b])
        error_list.append(abs(V_Y[b] - E_data2[b]))
    avg_error1 = sum / len(E_data2)
    print("Average Error on Validation Data: ", avg_error1)


linear_regression()


