import gerrychain
import geopandas as gpd
import functools
import numpy as np 
import pandas as pd
import tqdm
import scipy.stats as ss
import sklearn as skl
import json
import matplotlib.pyplot as plt



from gerrychain import Graph, Partition, Election, GeographicPartition
from gerrychain.updaters import Tally, cut_edges
from gerrychain import MarkovChain
from gerrychain.constraints import contiguous
from gerrychain.proposals import *
from gerrychain.accept import always_accept
from gerrychain.proposals import recom
from functools import partial
from gerrychain.metrics import mean_median
from gerrychain.metrics import partisan_bias
from gerrychain.metrics import partisan_gini
from gerrychain.metrics import efficiency_gap
from gerrychain.metrics import polsby_popper
from gerrychain.metrics import wasted_votes

from multiprocessing import Pool
import random

print("Reading Shapefile......")
shape = gpd.read_file("../data/PA_init/PA_VTD.shp")
print("Shapefile read!!!")

m = 9

def pp(plan):
    polsby = polsby_popper(plan)
    popper = 0
    for i in polsby: 
        popper += polsby[i]
    return popper/len(polsby)

def republican_constraint(partition):
    global m 
    if partition["SEN12"].wins("Rep") < m: 
        return False
    m = partition["SEN12"].wins("Rep")
    return True 

def chain(iterations):
    idef = random.randint(1, 10000)
    graph = Graph.from_json("../data/PA_init/PA_VTD.json")

    election = Election("SEN12", {"Dem": "USS12D", "Rep": "USS12R"})

    initial_partition = GeographicPartition(
        graph,
        assignment="2011_PLA_1",
        updaters={
            "cut_edges": cut_edges,
            "population": Tally("TOT_POP", alias="population"),
            "SEN12": election
        }
    )

    ideal_population = sum(initial_partition["population"].values()) / len(initial_partition)

    # We use functools.partial to bind the extra parameters (pop_col, pop_target, epsilon, node_repeats)
    # of the recom proposal.

    proposal = partial(recom,
                       pop_col="TOT_POP",
                       pop_target=ideal_population,
                       epsilon=0.02,
                       node_repeats=2
                      )

    chain = MarkovChain(
            proposal=proposal,
            constraints=[],
            accept=contiguous,
            initial_state=initial_partition,
            total_steps=85*iterations + 17000
        )

    count = 0
    metrics = []
    boundary_nodes = []
    boundary_weighted = []
    for partition in chain.with_progress_bar(): 
        mm = mean_median(partition["SEN12"])
        p = pp(partition)
        bias = partisan_bias(partition["SEN12"])
        gini = partisan_gini(partition["SEN12"])
        gap = efficiency_gap(partition["SEN12"])
        cut = len(partition["cut_edges"])
        if count >= 85:
            if count % 85 == 0: 
                metrics.append((mm, p, bias, gini, gap, cut, partition["SEN12"].wins("Rep")))
                nodes = [0]*8921
                bnodes = [0]*8921
                for edge in partition["cut_edges"]:
                    nodes[edge[0]] = 1
                    nodes[edge[1]] = 1
                    bnodes[edge[0]] += 1
                    bnodes[edge[1]] += 1
                boundary_nodes.append(nodes)
                boundary_weighted.append(bnodes)

                assign = {i: partition.assignment[i] for i in partition.assignment}
                shape["CD"] = shape.index.map(assign)
                this_map = shape.dissolve(by='CD')
                this_map.plot(color='black', edgecolor='white')

                plt.axis('off')

                plt.savefig("../images/PA_neutral/" + str(int((count-17000)/85)+1) + str(idef) + ".png")

                fig = plt.gcf()
                fig.set_size_inches((15,9), forward=False)
                fig.savefig("../images/PA_neutral/" + str(int((count-17000)/85)+1) + str(idef) + "r.png", dpi=600)

                plt.close()

            if count % 8500 == 0: 
                print(idef, count, mm, p, bias, gini, gap, cut, partition["SEN12"].wins("Rep"))
        else:
            if count%1000 == 0:
                print(idef, "Mixing...... Iteration", count, "/17000")
        count += 1

    return metrics, boundary_nodes, boundary_weighted

def gop_chain(iterations):
    idef = random.randint(1, 10000)
    graph = Graph.from_json("../data/PA_init/PA_VTD.json")

    election = Election("SEN12", {"Dem": "USS12D", "Rep": "USS12R"})

    initial_partition = GeographicPartition(
        graph,
        assignment="2011_PLA_1",
        updaters={
            "cut_edges": cut_edges,
            "population": Tally("TOT_POP", alias="population"),
            "SEN12": election
        }
    )

    ideal_population = sum(initial_partition["population"].values()) / len(initial_partition)

    # We use functools.partial to bind the extra parameters (pop_col, pop_target, epsilon, node_repeats)
    # of the recom proposal.

    proposal = partial(recom,
                       pop_col="TOT_POP",
                       pop_target=ideal_population,
                       epsilon=0.02,
                       node_repeats=2
                      )

    chain = MarkovChain(
            proposal=proposal,
            constraints=[republican_constraint],
            accept=contiguous,
            initial_state=initial_partition,
            total_steps=85*iterations + 17000
        )

    count = 0
    metrics = []
    boundary_nodes = []
    boundary_weighted = []
    for partition in chain.with_progress_bar(): 
        mm = mean_median(partition["SEN12"])
        p = pp(partition)
        bias = partisan_bias(partition["SEN12"])
        gini = partisan_gini(partition["SEN12"])
        gap = efficiency_gap(partition["SEN12"])
        cut = len(partition["cut_edges"])
        if count >= 17000:
            if count % 85 == 0:
                metrics.append((mm, p, bias, gini, gap, cut, partition["SEN12"].wins("Rep")))
                nodes = [0]*8921
                bnodes = [0]*8921
                for edge in partition["cut_edges"]:
                    nodes[edge[0]] = 1
                    nodes[edge[1]] = 1
                    bnodes[edge[0]] += 1
                    bnodes[edge[1]] += 1
                boundary_nodes.append(nodes)
                boundary_weighted.append(bnodes)

                assign = {i: partition.assignment[i] for i in partition.assignment}
                shape["CD"] = shape.index.map(assign)
                this_map = shape.dissolve(by='CD')
                this_map.plot(color='black', edgecolor='white')

                plt.axis('off')

                plt.savefig("../images/PA_gop/" + str(int((count-17000)/85)+1) + str(idef) + ".png")

                fig = plt.gcf()
                fig.set_size_inches((15,9), forward=False)
                fig.savefig("../images/PA_gop/" + str(int((count-17000)/85)+1) + str(idef) + "r.png", dpi=600)

                plt.close()

            if count % 8500 == 0: 
                print(idef, count, mm, p, bias, gini, gap, cut, partition["SEN12"].wins("Rep"))
        else:
            if count%1000 == 0:
                print(idef, "Mixing...... Iteration", count, "/17000")
        count += 1

    return metrics, boundary_nodes, boundary_weighted



pool = Pool(processes = 24)
N = 51000
results = pool.map(chain, (N/24, N/24, N/24, N/24, N/24, N/24, N/24, N/24, 
                           N/24, N/24, N/24, N/24, N/24, N/24, N/24, N/24, 
                           N/24, N/24, N/24, N/24, N/24, N/24, N/24, N/24))

pool = Pool(processes = 24)
N = 51000
gop_results = pool.map(gop_chain, (N/24, N/24, N/24, N/24, N/24, N/24, N/24, N/24, 
                               N/24, N/24, N/24, N/24, N/24, N/24, N/24, N/24, 
                               N/24, N/24, N/24, N/24, N/24, N/24, N/24, N/24))

metrics = np.concatenate([results[i][0] for i in range(24)])
boundary_nodes = np.concatenate([results[i][1] for i in range(24)])
boundary_weighted = np.concatenate([results[i][2] for i in range(24)])

df = pd.DataFrame(metrics)
df.columns = ["Mean-Median", "Polsby-Popper", "Bias", "Gini", "Gap", "Cuts", "Wins"]

df.to_csv("PA_NC_50000_20190809")

df2 = pd.DataFrame(boundary_nodes)
df2.to_csv("PA_BN_50000_20190809")

df3 = pd.DataFrame(boundary_weighted)
df3.to_csv("PA_BW_50000_20190809")

gmetrics = np.concatenate([results[i][0] for i in range(24)])
gboundary_nodes = np.concatenate([results[i][1] for i in range(24)])
gboundary_weighted = np.concatenate([results[i][2] for i in range(24)])

df = pd.DataFrame(gmetrics)
df.columns = ["Mean-Median", "Polsby-Popper", "Bias", "Gini", "Gap", "Cuts", "Wins"]

df.to_csv("PA_GOP_50000_20190809")

df2 = pd.DataFrame(gboundary_nodes)
df2.to_csv("PA_GOPBN_50000_20190809")

df3 = pd.DataFrame(gboundary_weighted)
df3.to_csv("PA_GOPBW_50000_20190809")
