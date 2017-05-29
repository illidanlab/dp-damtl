# 10 fold cross valiation to determine Lambda in Initialization.py

from numpy import dot, array, zeros, empty_like, delete
from math import log, exp
from numpy.linalg import norm
from scipy.optimize import minimize # it seems like the Java can't call this method
import csv
import matplotlib.pyplot as plt
from copy import deepcopy

def lr(z):
    # logistic loss
    logr = log(1+exp(-z))
    return logr

def obj(x): # x here is equal to q, where in the first iteration, p=0
    jfd = lr(labelc_train[0] * dot(data_train[0], x))
    for i in range(1, L):
        jfd = jfd + lr(labelc_train[i] * dot(data_train[i], x))
    f = (1.0 / L) * jfd + (Lambda / 2.0) * (norm(x) ** 2) # very careful about division in python, use 1.0 (flost), not 1 (int)
    return f

def split_data(dataset, num, size):
    # split the data point in dataset from num to num+size as the testing data and rest as the training data
    dataset_deepcopy = deepcopy(dataset)
    data_test = []
    for i in range(size):
        data_test.append(dataset_deepcopy[num+i])
    for i in range(size):
        del dataset_deepcopy[num] # the data point after deleted dataset_deepcopy(num) will automatically move ahead, so no displacement is needed
    data_train = dataset_deepcopy

    return data_train, data_test

def split_label(labelset, num, size):
    # simiar as split_data, but split label
    labelset_copy = empty_like(labelset)
    labelset_copy[:] = labelset
    label_test = empty_like(labelset_copy[num:num+size])
    label_test[:] = labelset_copy[num:num+size]
    index = []
    for i in range(size):
        index.append(num+i)
    label_train = delete(labelset_copy, index)

    return label_train, label_test

def err(data, labels, model):
    '''compute error rate'''
    e = 0.0 #number of errors
    n = len(labels)
    for i in range(n):
        sign = labels[i]*dot(model,data[i])
        if sign<0:
            e = e + 1.0
    res = 1.0*(e/n)
    return res

data = [] # data for current task
labelc = [] # label for current task (classification problem)

with open('data' + str(1), 'rb') as f:
    try:
        reader = csv.reader(f)
        for row in reader:
            data.append(row[0].split(' '))  # extract 'string' from ['string'], split it by ' ' and store them in []
    finally:
        f.close()
data.pop(0)  # remove first one
for i in data:
    del i[-1]  # remove last one, which is a ''
    for j in range(len(i)):
        i[j] = float(i[j])  # string to float number
    i = array(i)
D = len(data[0]) # type(data): list, type(data[0]): list

with open('labelc' + str(1), 'rb') as f:
    try:
        reader = csv.reader(f)
        for row in reader:
            labelc.append(float(row[0][:-1]))  # extract 'string' from ['string'], remove last ' '
    finally:
        f.close()
labelc.pop(0)
labelc = array(labelc) # type(labelc): numpy.ndarray, type(labelc[0]): numpy.float64
L = len(labelc)

KFOLD = 10 # number of fold in cross valiation
s = int(L/KFOLD) # number of data point in each fold
L = L - s
x0 = zeros(D)  # starting point with same length as any data point
data_train = [] # training data for current fold
labelc_train = [] # training label for current fold
error_all = [] # error rates for all value of Lambda
count = 0 # starting index of data point in testing set

for k in range(20):
    Lambda = (10 ** -10)*(10 ** k)  # regularization parameter
    print "Lambda=" + str(Lambda)
    error = []  # error rates for fixed Lambda, len(error) = KFOLD
    for i in range(KFOLD):
        count = i*s
        data_train, data_test = split_data(data, count, s)
        labelc_train, labelc_test = split_label(labelc, count, s) # training label for current task (classification problem)
        w = minimize(obj, x0, method='Nelder-Mead').x # minimization procedure
        print w
        error.append(err(data_test, labelc_test, w))
    error_all.append(reduce(lambda x, y: x + y, error) / len(error))

print error_all

t = range(len(error_all))
plt.plot(t, error_all, 'ro')
plt.show()