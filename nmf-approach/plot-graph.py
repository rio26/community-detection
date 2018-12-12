import networkx as nx
import matplotlib.pyplot as plt
from time import time
from sklearn import cluster
from collections import defaultdict
import matplotlib.pyplot as plt
from matplotlib import cm, colors
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.metrics.cluster import normalized_mutual_info_score
from sklearn.metrics.cluster import adjusted_rand_score


G = nx.karate_club_graph()


def draw_communities(G, membership, pos):
    """Draws the nodes to a plot with assigned colors for each individual cluster
    Parameters
    ----------
    G : networkx graph
    membership : list
        A list where the position is the student and the value at the position is the student club membership.
        E.g. `print(membership[8]) --> 1` means that student #8 is a member of club 1.
    pos : positioning as a networkx spring layout
        E.g. nx.spring_layout(G)
    """ 
    fig, ax = plt.subplots(figsize=(16,9))
    
    # Convert membership list to a dict where key=club, value=list of students in club
    club_dict = defaultdict(list)
    for student, club in enumerate(membership):
        club_dict[club].append(student)
    
    # Normalize number of clubs for choosing a color
    norm = colors.Normalize(vmin=0, vmax=len(club_dict.keys()))
    
    for club, members in club_dict.items():
        nx.draw_networkx_nodes(G, pos,
                               nodelist=members,
                               node_color=cm.jet([norm(club)]),
                               node_size=500,
                               alpha=0.8,
                               ax=ax)

    # Draw edges (social connections) and show final plot
    plt.title("Zachary's Karate Club")
    nx.draw_networkx_edges(G, pos, alpha=0.5, ax=ax)

membership = [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
pos = nx.spring_layout(G)

draw_communities(G, membership, pos)
plt.draw()
plt.show()

############### plot 2 ###################
'''
G = nx.read_gml('dolphins-v62-e159/dolphins.gml') # 62 vertices
#### OR ####
# G = nx.Graph()
# with open('email-v1005-e25571-c42/email-Eu-core.txt','r') as f:
# # with open('small-test-graphs/v15-c3.txt','r') as f:
#     for line in f:
#         line=line.split()#split the line up into a list - the first entry will be the node, the others his friends
#         # print(len(line), "cont", line[0])
#         if len(line)==1:#in case the node has no friends, we should still add him to the network
#             if line[0] not in G:
#                 nx.add_node(line[0])
#         else:#in case the node has friends, loop over all the entries in the list
#             focal_node = line[0]#pick your node
#             # print(line[1:])
#             for friend in line[1:]:#loop over the friends
#                 if friend != focal_node:
#                     G.add_edge(focal_node,friend)#add each edge to the graph
############

# cluster_num = 3
t0 = time()
nx.draw(G, with_labels=True)
t1 = time()


plt.draw()
plt.show()

print("time taken: ," , t1-t0)
'''
######################################