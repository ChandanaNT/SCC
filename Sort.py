with open('./PP_Project/DataSets/LiveJournal.txt') as f:
    lines = f.readlines()

#Check number of edges
#print(len(lines))

edge_list = []

for edge in lines:
    #Remove '/n' at the end of every line
    edge = edge[0:len(edge)-1]
    nodes = edge.split('\t')
    #print(nodes)
    frome = nodes[0]
    toe = nodes[1]
    edge_list.append([frome,toe])
    

#print(edge_list)
edge_list.sort(key=lambda x: x[0])
#print(edge_list)

thefile = open('sortedLiveJournal.txt', 'w')
for item in edge_list:
  thefile.write("%s %s\n" % (item[0],item[1]))
