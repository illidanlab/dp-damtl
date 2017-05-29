#Implememt java code in /home/decs/Desktop/javaworkspace/project/src/Initialization.java in Python.
#Initialize $\q_{t}^{(0)}$ in each task node using STL, $\S^{(0)} = \P^{(0)}=0_{d\times T}$ on central server side.
#comment write file code to debug

from numpy import array, dot, zeros
from math import log, exp
from numpy.linalg import norm
from scipy.optimize import minimize # it seems like the Java can't call this method
import csv

def lr(z):
    #logistic loss
    logr = log(1+exp(-z))
    return logr

def obj(x): # x here is equal to q, where in the first iteration, p=0, classification
    jfd = lr(label_train[0] * dot(data_train[0], x))
    for i in range(1, L):
        jfd = jfd + lr(label_train[i] * dot(data_train[i], x))
    f = (1.0 / L) * jfd + (Lambda / 2.0) * (norm(x) ** 2) # very careful about division in python, use 1.0 (flost), not 1 (int)
    return f
'''
def obj(x): # x here is equal to q, where in the first iteration, p=0, regression
    jfd = (dot(data_train[0], x) - label_train[0])**2
    for i in range(1, L):
        jfd = jfd + (dot(data_train[i], x) - label_train[i])**2
    f = (1.0 / L) * jfd + (Lambda / 2.0) * (norm(x) ** 2) # very careful about division in python, use 1.0 (flost), not 1 (int)
    return f
'''
data = [] # data for current task
data_train = [] # training data for current task
label = [] # label for current task
label_train = [] # training label for current task
tasks = [] # q for all tasks
P = [] # initial value of shared component
S = [] # initial value of gradient matrix (w.r.t. p) at central server
T = 20 # number of task
L = 0 # number of data point in current task
D = 0 # data dimension
p_train = 0.3 # percentage of training set
Lambda = 10 ** -3 # regularization parameter
x0 = zeros(D)  # starting point with same length as any data point

for k in range(T):
    with open('data'+str(k+1), 'rb') as f:
        try:
            reader = csv.reader(f)
            for row in reader:
                data.append(row[0].split(' ')) # extract 'string' from ['string'], split it by ' ' and store them in []
        finally:
            f.close()
    data.pop(0) # remove first one

    for i in data:
        del i[-1] # remove last one, which is a ''
        for j in range(len(i)):
            i[j] = float(i[j]) # string to float number
        i = array(i)
    D = len(data[0])
    x0 = zeros(D)

    with open('labelc'+str(k+1), 'rb') as f:
    #with open('label' + str(k + 1), 'rb') as f:
        try:
            reader = csv.reader(f)
            for row in reader:
                label.append(float(row[0][:-1])) # extract 'string' from ['string'], remove last ' '
        finally:
            f.close()
    label.pop(0)
    label = array(label)
    L = len(label)
    L = int(L*p_train)
    data_train = data[0:int(len(data)*p_train)]# get training data and label
    label_train = label[0:int(len(label)*p_train)]
    w = minimize(obj, x0, method='Nelder-Mead').x # minimization procedure
    tasks.append(w)
    data = [] # re-initialization
    data_train = []
    label = []
    label_train = []

P = zeros((D, T)) # create all zero matrix the same shape as tasks
S = zeros((D, T))

f1 = open('/home/decs/Desktop/Javaws/DAMTLDP/startQ', 'wt') # create and write into startQ file in java workspace
f2 = open('/home/decs/Desktop/Javaws/DAMTLDP/startP', 'wt') # create and write into startP file in java workspace
f3 = open('/home/decs/Desktop/Javaws/DAMTLDP/startS', 'wt') # create and write into startS file in java workspace

try:
    writer1 = csv.writer(f1, delimiter=' ') # use delimiter=' ' to avoid ',' as delimiter
    writer2 = csv.writer(f2, delimiter=' ') 
    writer3 = csv.writer(f3, delimiter=' ') 
    writer1.writerow((T, D))
    writer2.writerow((D, T))
    writer3.writerow((D, T))
    for i in range(T):
        writer1.writerow(tasks[i])
    for j in range(D):
        writer2.writerow(P[j])
        writer3.writerow(S[j])
finally:
    f1.close()
    f2.close()
    f3.close()




