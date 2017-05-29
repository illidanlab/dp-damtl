from numpy import dot, array
from copy import deepcopy
import matplotlib.pyplot as plt
from random import randint
import csv
import scipy.io

mat = scipy.io.loadmat('school.mat')
#both mat['X'][0] and mat['Y'][0] have 139 elements, hence contain the data and score, both types are <type 'numpy.ndarray'>, shape is n*d
D = len(mat['X'][0][0][0])
NT = 20 # number of tasks


for i in range(NT):
    t = mat['X'][0][i]
    T = len(mat['X'][0][i])
    f1 = open('/home/decs/Desktop/Javaws/DAMTLDP/schoolCSVdata' + str(i + 1), 'wt')
    writer1 = csv.writer(f1, delimiter=' ')
    writer1.writerow((T, D))
    for row in t:
        writer1.writerow(row)
    s = mat['Y'][0][i]
    f2 = open('/home/decs/Desktop/Javaws/DAMTLDP/schoolCSVscore' + str(i + 1), 'wt')
    writer2 = csv.writer(f2, delimiter=' ')
    writer2.writerow((T, 1))
    for row in s:
        writer2.writerow(row)

