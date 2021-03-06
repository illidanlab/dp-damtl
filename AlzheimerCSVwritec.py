import pandas as pd
import csv

file_name = "Alzheimer.xlsx" # yellow part as feature, normalize before use
D = 248 # data dimension
NT = 20 # number of tasks

label = pd.read_excel(file_name, index_col=None, na_values=['NA'], parse_cols = "D") # extract column D, DX_bl (CN(normal) and AD(Alzheimer)) for classification
data = pd.read_excel(file_name, index_col=None, na_values=['NA'], parse_cols = "X:JK") # extract from column X to column JK, features
id = pd.read_excel(file_name, index_col=None, na_values=['NA'], parse_cols = "A") # extract column A, SUBJECT_ID for ID

labell = [] # store label, type(labell[0]): <type 'unicode'>, type(labell): <type 'list'>
for i in label.index:
    labell.append(label['DX_bl'][i])

idl = [] # store id
for i in id.index:
    idl.append(id['SUBJECT_ID'][i]) # len(idl): 1827

datal = [] # store data, each row is a data point
for i in data.index:
    temp = []  # one feature
    for j in range(D): # type(j): numpy.int64
        temp.append(data.iloc[i][j])
    datal.append(temp)

# discard data point contains non-numerical value, and corresponding label/id
Index = [] # remember the index of data point which has non-numerical feature
for i in range(len(datal)):
    if datal[i][0] == '.' :
        Index.append(i)

for i in sorted(Index, reverse=True):
    del labell[i] # all with length 1688
    del idl[i]
    del datal[i]

# discard label contains value other than 'CN' or 'AD', and corresponding data/id
Index = []
for i in range(len(labell)):
    if labell[i] == 'CN' or labell[i] == 'AD' :
        pass
    else:
        Index.append(i)

for i in sorted(Index, reverse=True):
    del labell[i] # all with length 746
    del idl[i]
    del datal[i]

# label transformation
for i in range(len(labell)):
    if labell[i] == 'CN': # +1 for normal people
        labell[i] = +1
    else:
        labell[i] = -1

# task seperation, according to first 3 digits of SUBJECT_ID
ID_dict = dict() # dict of possible task id with corresponding task
ID_tar_dict = dict() # dict of possible task id with corresponding targets
for i in range(len(idl)): # type(idl[i]): <type 'unicode'>
    if idl[i][:3] in ID_dict:
        pass
    else:
        ID_dict[idl[i][:3]] = [] # prepare to append data point to each task
        ID_tar_dict[idl[i][:3]] = []

# add data point and label to each task
for i in range(len(idl)):
    ID_dict[idl[i][:3]].append(datal[i])
    ID_tar_dict[idl[i][:3]].append(labell[i])

# select top 20 tasks with largest number of data points
l = []
for i in ID_dict.keys():
    l.append(len(ID_dict[i]))
ls = sorted(l, reverse=True) # sort in decreasing order
thr = ls[19]
Data = [] # store data
Label = [] # store labels
Row = [] # store corresponding number of data
count = 0 # count the number of data set already selected, since we use the threshold to select data sets
for i in ID_dict.keys():
    if len(ID_dict[i]) >= thr:
        Data.append(ID_dict[i])
        Label.append(ID_tar_dict[i])
        Row.append(len(ID_dict[i]))
        count = count + 1
    if count == NT:
       break

# write into files
if len(Row) == NT:
    print 'Everthing is fine'
else:
    print 'Something is wrong'

for i in range(NT):
    t = Data[i]
    T = Row[i]
    f1 = open('/home/decs/Desktop/Javaws/DAMTLDP/AlzheimerCSVdatac' + str(i + 1), 'wt')
    writer1 = csv.writer(f1, delimiter=' ')
    writer1.writerow((T, D))
    for row in t:
        writer1.writerow(row)
    s = Label[i]
    f2 = open('/home/decs/Desktop/Javaws/DAMTLDP/AlzheimerCSVlabel' + str(i + 1), 'wt')
    writer2 = csv.writer(f2, delimiter=' ')
    writer2.writerow((T, 1))
    for row in s:
        writer2.writerow([row])






