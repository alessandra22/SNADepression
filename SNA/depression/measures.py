import copy
import numpy as np
import pandas as pd
import networkx as nx
from matplotlib import pyplot as plt
from matplotlib import patches as pat
from scipy.stats import pearsonr
from sklearn.linear_model import LinearRegression
from SNA.depression.loadData import dat
from SNA.depression.loadData import g1, g2

metrics = {'depression.L1': 'depression score', 'n.sec.h': 'time spent in interaction',
           'ratio': 'ratio dyadic/group interaction', 'depression.zone': 'depression zone'}


def f2(val):  # format val in order to have only two decimal values
    return "{:.2f}".format(val)


def f3(val):  # format val in order to have only three decimal values
    return "{:.3f}".format(val)


def get_lr(x, y_train):  # compute the linear regression
    x_train = np.array(x).reshape(-1, 1)
    model = LinearRegression()
    model.fit(x_train, y_train)
    return x_train, model.predict(x_train)


def print_correlation_coefficient_attributes(metric1, metric2):
    def get_correlation_coefficient_attributes(g=None):
        if g:
            dat1 = nx.get_node_attributes(g, metric1)
            dat2 = nx.get_node_attributes(g, metric2)
            new_df = pd.DataFrame({metric1: dat1, metric2: dat2}).dropna().sort_index()
        else:
            df = pd.DataFrame(dat).sort_index()
            new_df = pd.DataFrame({metric1: df[metric1], metric2: df[metric2]}).dropna().sort_index()

        return {'cc': np.corrcoef([new_df[metric1], new_df[metric2]])[0, 1],
                'p-value': pearsonr(new_df[metric1], new_df[metric2]).pvalue,
                'df': new_df}

    title = f"Correlation between {metrics[metric1]} and {metrics[metric2]}"
    cc_g1 = get_correlation_coefficient_attributes(g1)
    cc_g2 = get_correlation_coefficient_attributes(g2)
    cc = get_correlation_coefficient_attributes()

    x_train1, y_pred1 = get_lr(cc_g1['df'][metric1], cc_g1['df'][metric2].values)
    x_train2, y_pred2 = get_lr(cc_g2['df'][metric1], cc_g2['df'][metric2].values)
    x_train_glob, y_pred_glob = get_lr(cc['df'][metric1], cc['df'][metric2].values)

    plt.plot(cc_g1['df'][metric1].values, cc_g1['df'][metric2].values, 'rs', label='net 1')
    plt.plot(cc_g2['df'][metric1].values, cc_g2['df'][metric2].values, 'b^', label='net 2')
    plt.plot(x_train1, y_pred1, 'r', x_train2, y_pred2, 'b', x_train_glob, y_pred_glob, 'g')
    plt.xlabel(metrics[metric1])
    plt.ylabel(metrics[metric2])
    plt.legend(loc="lower right")
    plt.title(title)
    plt.draw()
    plt.show()

    print(title, ':\n',
          '\tG1 = ', f2(cc_g1['cc']), '\twith a p-value of', f3(cc_g1['p-value']), '\n',
          '\tG2 = ', f2(cc_g2['cc']), '\twith a p-value of', f3(cc_g2['p-value']), '\n',
          '\tglobal = ', f2(cc['cc']), '\twith a p-value of', f3(cc['p-value']), '\n')


def print_correlation_coefficient_centrality(metric):
    def get_df_attribute_centrality(g):
        node_attr = nx.get_node_attributes(g, metric)
        centrality = nx.degree_centrality(g)
        return pd.DataFrame({'degree centrality': centrality.values(), metric: node_attr.values()},
                            index=list(centrality.keys())).dropna().sort_index()

    def get_correlation_coefficient_centrality_df(df):
        return {'cc': np.corrcoef([df[metric], df['degree centrality']])[0, 1],
                'p-value': pearsonr(df[metric], df['degree centrality']).pvalue,
                'df': df}

    title = f"Correlation between {metrics[metric]} and degree centrality"
    cc_g1 = get_correlation_coefficient_centrality_df(get_df_attribute_centrality(g1))
    cc_g2 = get_correlation_coefficient_centrality_df(get_df_attribute_centrality(g2))

    x_train1, y_pred1 = get_lr(cc_g1['df'][metric], cc_g1['df']['degree centrality'].values)
    x_train2, y_pred2 = get_lr(cc_g2['df'][metric], cc_g2['df']['degree centrality'].values)

    plt.plot(cc_g1['df'][metric].values, cc_g1['df']['degree centrality'].values, 'rs', label='net 1')
    plt.plot(cc_g2['df'][metric].values, cc_g2['df']['degree centrality'].values, 'b^', label='net 2')
    plt.plot(x_train1, y_pred1, 'r', x_train2, y_pred2, 'b')
    plt.draw()
    plt.xlabel(metrics[metric])
    plt.ylabel('degree centrality')
    plt.legend(loc="lower right")
    plt.title(title)
    plt.show()

    print(title, '\n',
          '\tG1 = ', f2(cc_g1['cc']), '\twith a p-value of', f3(cc_g1['p-value']), '\n',
          '\tG2 = ', f2(cc_g2['cc']), '\twith a p-value of', f3(cc_g2['p-value']), '\n')


def print_homophily_for_metric(metric):
    print('Homophily in network 1 for', metrics[metric], ':', f3(nx.attribute_assortativity_coefficient(g1, metric)))
    print('Homophily in network 2 for', metrics[metric], ':', f3(nx.attribute_assortativity_coefficient(g2, metric)))
    print()


def print_interaction_time_zones():
    def get_color(val):
        if val == 'l':
            return 'b'
        if val == 'c':
            return 'r'
        else:
            return 'orange'

    def plot_and_return_mean_time(g, network):
        edges = g.edges
        edges_dict = dict()
        for e in edges:
            edges_dict[e[0], e[1]] = edges[e]['weight']

        nodes = g.nodes
        nodes_dict = dict()
        for n in nodes.keys():
            nodes_dict[n] = dict()
            nodes_dict[n]['same'] = 0
            nodes_dict[n]['different'] = 0
            nodes_dict[n]['l'] = 0
            nodes_dict[n]['sc'] = 0
            nodes_dict[n]['c'] = 0

        for e in edges_dict.keys():
            if nodes[e[0]]['depression.zone'] == nodes[e[1]]['depression.zone']:
                nodes_dict[e[0]]['same'] += edges_dict[e]
                nodes_dict[e[1]]['same'] += edges_dict[e]
            else:
                nodes_dict[e[0]]['different'] += edges_dict[e]
                nodes_dict[e[1]]['different'] += edges_dict[e]

            if nodes[e[0]]['depression.zone'] == 'c':
                nodes_dict[e[1]]['c'] += edges_dict[e]
            elif nodes[e[0]]['depression.zone'] == 'sc':
                nodes_dict[e[1]]['sc'] += edges_dict[e]
            else:
                nodes_dict[e[1]]['l'] += edges_dict[e]

            if nodes[e[1]]['depression.zone'] == 'c':
                nodes_dict[e[0]]['c'] += edges_dict[e]
            elif nodes[e[1]]['depression.zone'] == 'sc':
                nodes_dict[e[0]]['sc'] += edges_dict[e]
            else:
                nodes_dict[e[0]]['l'] += edges_dict[e]

        total_time = {'l': {'l': 0, 'sc': 0, 'c': 0}, 'sc': {'l': 0, 'sc': 0, 'c': 0}, 'c': {'l': 0, 'sc': 0, 'c': 0}}
        tot_zone = {'l': 0, 'sc': 0, 'c': 0}

        for n in nodes_dict.keys():
            total_time[nodes[n]['depression.zone']]['l'] += nodes_dict[n]['l']
            total_time[nodes[n]['depression.zone']]['sc'] += nodes_dict[n]['sc']
            total_time[nodes[n]['depression.zone']]['c'] += nodes_dict[n]['c']
            tot_zone[nodes[n]['depression.zone']] += 1

        mean_time = total_time
        for k in mean_time.keys():
            for z in mean_time[k].keys():
                mean_time[k][z] = total_time[k][z] / tot_zone[k]

        weighted_mean_time = copy.deepcopy(mean_time)
        my_ticks = ['lack', 'subclinical', 'critical']
        plt.xlabel('depression zone')
        plt.ylabel('mean time spent')
        plt.title(f'Mean weighted time of interaction for network {network}')
        plt.xticks([1, 3, 5], my_ticks)
        i = 1
        for k in weighted_mean_time.keys():
            for z in weighted_mean_time[k].keys():
                weighted_mean_time[k][z] = mean_time[k][z] / tot_zone[z]
                plt.plot(i, weighted_mean_time[k][z], color=get_color(z), marker='o')
            i += 2

        patches = [pat.Patch(color='blue', label='lack'), pat.Patch(color='orange', label='subclinical'),
                   pat.Patch(color='red', label='critical')]
        plt.legend(handles=patches, loc='lower left')
        plt.draw()
        plt.show()
        return mean_time, tot_zone, weighted_mean_time

    def print_dict_zones(title, d, tot_z):
        print(f"\n{title}")
        for k in mt1:
            print(f"{k} ({tot_z[k]})\t--->  l =", f2(d[k]['l']), "- sc =", f2(d[k]['sc']), "- c =", f2(d[k]['c']))

    mt1, tz1, wm1 = plot_and_return_mean_time(g1, '1')
    mt2, tz2, wm2 = plot_and_return_mean_time(g2, '2')

    # print_dict_zones('Mean time of interaction for network 1:', mt1, tz1)
    print_dict_zones('Mean weighted time of interaction for network 1:', wm1, tz1)
    # print_dict_zones('Mean time of interaction for network 2:', mt2, tz2)
    print_dict_zones('Mean weighted time of interaction for network 2:', wm2, tz2)


print_functions = {'correlation coefficient between attributes': print_correlation_coefficient_attributes,
                   'correlation coefficient between attribute and degree centrality':
                       print_correlation_coefficient_centrality,
                   'homophily for a metric': print_homophily_for_metric,
                   'mean time spent and zones': print_interaction_time_zones}
