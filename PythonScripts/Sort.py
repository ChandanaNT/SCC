#Script used to sort the edges in the directed graph by 1st vertex(from vertex) number

filename = 'wiki-Vote'
with open(filename + '.txt') as f:
    lines = f.readlines()

#Check number of edges
#print(len(lines))

edge_list = []

for edge in lines:
    #Remove '/n' at the end of every line
    edge = edge[0:len(edge)-1]
    nodes = edge.split()
    if(len(nodes) == 1 or len(nodes) == 0):
        break
    frome = nodes[0]
    toe = nodes[1]
    edge_list.append([frome,toe])



edge_list.sort(key=lambda x: int(x[0]))
#print(edge_list)

thefile = open(filename + '_Sorted.txt', 'w')
for item in edge_list:
 thefile.write("%s %s\n" % (item[0],item[1]))
