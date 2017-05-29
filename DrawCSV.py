#Plot data stored in .csv file 
import csv
import sys
from numpy import arange
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pylab import *

csv.field_size_limit(sys.maxsize) # for huge file
L = []
error = [] # error rate of all tasks
T = 20 # number of tasks

'''
# plot multiple figures from multiple files

for i in range(T):
    e = [] # error rate (or MSE) of one task
    with open('/home/decs/Desktop/2017KDDResult/Exp1CAlzDP0540/errorRate'+str(i+1), 'rb') as f:
        reader = csv.reader(f)
        L = list(reader) # L[1][0] contains error rate data, in string form
    for j in L[1][0].split():
        e.append(float(j))
    error.append(e)
    e = []

l = 10000
ite = arange(l) # number of iterations

# truncate each err to same length, if the task have different iteration
for i in range(T):
    error[i] = error[i][0:l]

for i in range(T):
    plt.plot(ite, error[i], 'r--')

red_patch = mpatches.Patch(color='red', label='Testing error rate')
#red_patch = mpatches.Patch(color='red', label='Mean square error')
#red_patch = mpatches.Patch(color='blue', label='Weighted MSE')
plt.legend(handles=[red_patch])
plt.xlabel('Iterations')
#plt.ylabel('Mean square error in each task node')
plt.ylabel('Classification error rate in each task node')
#plt.xlim(0,500)
plt.show()


# plot one figure
err = []
with open('/home/decs/Desktop/2017KDDResult/Exp1CAlzDP0560/objvalue', 'rb') as f:
    reader = csv.reader(f)
    L = list(reader) # L[1][0] contains error rate data, in string form

for j in L[1][0].split():
    err.append(float(j))
#for i in range(1000): # only for file Exp1CAlzDP0540
#   err[199000-1+i] = err[198000-1+i]
ite = arange(len(err))
plt.plot(ite, err, 'b--')
plt.xlabel('Iteration (asynchronous)')
#plt.xlim(0,200)
plt.show()

'''
# plot multiple objective function values from multiple files

num = 6 # number of figures
objvalue = [] # objective function value, for different parameter value
for i in range(num):
    value = []  # objective function value, for one parameter value
    with open('/home/decs/Desktop/2017KDDResult/test/objvalue'+str(i+1), 'rb') as f:
        reader = csv.reader(f)
        L = list(reader) # L[1][0] contains error rate data, in string form
    for j in L[1][0].split():
        value.append(float(j))
    objvalue.append(value)
    value = []

l = 200000
ite = arange(l) # number of iterations

axis_font = {'size': '15', 'weight': 'bold'}
ax = gca()
fontsize = 15
for tick in ax.xaxis.get_major_ticks():
    tick.label1.set_fontsize(fontsize)
    tick.label1.set_fontweight('bold')
for tick in ax.yaxis.get_major_ticks():
    tick.label1.set_fontsize(fontsize)
    tick.label1.set_fontweight('bold')

g, = plt.plot(ite, objvalue[0][:l], 'g--') # lw="4", adjust weight of line
r, = plt.plot(ite, objvalue[1][:l], 'r--')
c, = plt.plot(ite, objvalue[2][:l], 'c--')
m, = plt.plot(ite, objvalue[3][:l], 'm--')
y, = plt.plot(ite, objvalue[4][:l], 'y--')
b, = plt.plot(ite, objvalue[5][:l], 'b--')
#plt.legend([g, r, c, m, y], ["Task 1", "Task 2", "Task 3", "Task 4", "Task 5"], prop={'weight':'bold'}) # for each task node
#plt.legend([g, r, c, m, y, b], ["eps=3.0", "eps=4.0", "eps=5.0", "eps=6.0", "eps=7.0", "non-DP"], prop={'weight':'bold'})
#plt.legend([g, r, c, m, y, b], ["eps=4.0", "eps=5.0", "eps=6.0", "eps=8.0", "eps=10.0", "non-DP"], prop={'weight':'bold'})
plt.legend([g, r, c, m, y, b], ["eps=0.3", "eps=0.4", "eps=0.5", "eps=1.0", "eps=2.0", "non-DP"], prop={'weight':'bold'})

#plt.xlabel('Iteration', **axis_font)
plt.xlabel('Iteration (asynchronous)', **axis_font)
#plt.ylabel('Classification error rate', **axis_font)
#plt.ylabel('Mean square error', **axis_font)
#plt.xlim(0,10000)
#plt.ylim(0,4.0)
plt.ylabel('Objective function value', **axis_font)
plt.show()
