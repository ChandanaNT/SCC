import random

filename = 'smallDummyData'
num_edges = 10
num_vertices = num_edges -  random.randint(int(num_edges/5),int(num_edges/3))
print(num_vertices)

thefile = open(filename + '.txt', 'w')

edges = 0
while(edges != num_edges):
    fromv = random.randint(0,num_vertices)
    tov = random.randint(0, num_vertices)
    if(fromv != tov):
         thefile.write("%s %s\n" % (fromv, tov))
         edges = edges + 1
        
