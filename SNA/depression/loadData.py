import numpy as np
import pandas as pd
import networkx as nx


def get_color_edges(val):
    if val > 4.5:
        return 'gray'
    else:
        return 'lightgrey'


def get_color_nodes(val):
    if val >= 20:
        return 'red'
    if val >= 15:
        return 'orange'
    if val >= 10:
        return 'yellow'
    if val >= 5:
        return 'green'
    return 'white'


# add a depression zone for calculating assortative mixing in the network
def depression_zone(val):
    if val > 16:
        return 'c'  # critical
    if val >= 9:
        return 'sc'  # subclinical
    return 'l'  # lack


# data loading
dat = pd.read_csv("../depression/data/dat.csv", delimiter=',', index_col='Unnamed: 0')
dvs1 = pd.read_csv("../depression/data/dvs1.csv", delimiter=',', index_col='Unnamed: 0')
dvs2 = pd.read_csv("../depression/data/dvs2.csv", delimiter=',', index_col='Unnamed: 0')

# data cleaning
dvs1.columns = dvs1.index  # 1-73 instead of X1-X73
dvs2.columns = dvs2.index  # 1-50 instead of X1-X50

# let's create a dictionary of dictionaries where the first key is the ID of a student,
# the dict value will then have many keys as the columns of the database
attributes1 = dict()
attributes2 = dict()

# we are keeping in the study only the students that took the test, then clean also dvs1 and dvs2
for i in range(1, 74):
    if not np.isnan(dat.loc[i]['depression.L1']):
        attributes1[i] = dict()
        for c in dat.columns:
            attributes1[i][c] = dat.loc[i][c]
        attributes1[i]['depression.zone'] = depression_zone(attributes1[i]['depression.L1'])

for i in range(74, 124):
    if not np.isnan(dat.loc[i]['depression.L1']):
        attributes2[i - 73] = dict()
        for c in dat.columns:
            attributes2[i - 73][c] = dat.loc[i][c]
        attributes2[i - 73]['depression.zone'] = depression_zone(attributes2[i - 73]['depression.L1'])

# we keep two graphs of all the students, even without depression scores in order
# to compute better interaction time later
g1_raw = nx.from_pandas_adjacency(dvs1)
g2_raw = nx.from_pandas_adjacency(dvs2)

# we clean the dvs1 database after applying the attributes because otherwise
# we wouldn't be able to assign the correct id to every student
for i in dvs1.index:
    if i not in attributes1.keys():
        dvs1 = dvs1.drop(i, axis=0)
        dvs1 = dvs1.drop(i, axis=1)

for i in dvs2.index:
    if i not in attributes2.keys():
        dvs2 = dvs2.drop(i, axis=0)
        dvs2 = dvs2.drop(i, axis=1)


# two different graph are created for the two adjacency matrices in dvs1.csv and dvs2.csv
# and with this function, edges are already weighted
g1 = nx.from_pandas_adjacency(dvs1)
g2 = nx.from_pandas_adjacency(dvs2)
colors = 'a'
weights = 'b'
# then the attributes in dat.csv are correctly assigned to the respective nodes
nx.set_node_attributes(g1, attributes1)
nx.set_node_attributes(g2, attributes2)
# two functions for debugging and be sure that nodes and edges have the correct data inside
# print(g1.nodes()[32])  # DEBUG
# print(g2.get_edge_data(66, 70))  # DEBUG

weight1 = nx.get_edge_attributes(g1, 'weight')
weight2 = nx.get_edge_attributes(g2, 'weight')

for e in g1.edges():
    w = nx.get_edge_attributes(g1, 'weight')[e]
    nx.set_edge_attributes(g1, {e: {'weight': w, 'color': get_color_edges(w)}})

for e in g2.edges():
    w = nx.get_edge_attributes(g2, 'weight')[e]
    nx.set_edge_attributes(g2, {e: {'weight': w, 'color': get_color_edges(w)}})

colors_edges1 = [g1[u][v]['color'] for u, v in g1.edges()]
weight1 = [g1[u][v]['weight'] for u, v in g1.edges()]
colors_edges2 = [g2[u][v]['color'] for u, v in g2.edges()]
weight2 = [g2[u][v]['weight'] for u, v in g2.edges()]

colors_nodes1 = [get_color_nodes(g1.nodes[i]['depression.L1']) for i in g1.nodes]
colors_nodes2 = [get_color_nodes(g2.nodes[i]['depression.L1']) for i in g2.nodes]

colors = {'edges1': colors_edges1, 'edges2': colors_edges2, 'nodes1': colors_nodes1, 'nodes2': colors_nodes2}
weights = {1: weight1, 2: weight2}
