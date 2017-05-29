# generate configuration file

f = open("configuration","w")
T = 20 # number of tasks
latency = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
print len(latency)
for i in range(T):
    f.write(str(i+1))
    f.write("\n")
    f.write(str(latency[i]))
    f.write("\n")
f.close()
