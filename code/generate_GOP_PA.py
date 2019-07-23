import gerrychain
import functools
import numpy as np 
import pandas as pd
import tqdm
import scipy.stats as ss
import sklearn as skl

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
    graph = Graph.from_json("./PA_VTD.json")

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
            total_steps=iterations + 100
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
        if count >= 100:
            metrics.append((mm, p, bias, gini, gap, cut))
            nodes = [0]*8921
            bnodes = [0]*8921
            for edge in partition["cut_edges"]:
                nodes[edge[0]] = 1
                nodes[edge[1]] = 1
                bnodes[edge[0]] += 1
                bnodes[edge[1]] += 1
            boundary_nodes.append(nodes)
            boundary_weighted.append(bnodes)
        if count % 100 == 0: 
            print(idef, count, mm, p, bias, gini, gap, cut, partition["SEN12"].wins("Rep"))
        count += 1

    return metrics, boundary_nodes, boundary_weighted

pool = Pool(processes = 24)
N = 51000
results = pool.map(chain, (N/24, N/24, N/24, N/24, N/24, N/24, N/24, N/24, 
                           N/24, N/24, N/24, N/24, N/24, N/24, N/24, N/24, 
                           N/24, N/24, N/24, N/24, N/24, N/24, N/24, N/24))

metrics = []
boundary_nodes = []
boundary_weighted = []
print("Compiling Data........")
for i in range(24):
    metrics.extend(results[i][0])
    boundary_nodes.extend(results[i][1])
    boundary_weighted.extend(results[i][2])
    print("Process " + str(i+1) + "/24.... DONE")


print("Writing Metrics........")
df = pd.DataFrame(metrics)
df.columns = ["Mean-Median", "Polsby-Popper", "Bias", "Gini", "Gap", "Cuts"]

df.to_csv("PA_GOP_50000_20190721")

print("Writing Boundary Nodes........")
df2 = pd.DataFrame(boundary_nodes)
df2.to_csv("PA_GOPBN_50000_20190721")

print("Writing Boundary Weighted........")
df3 = pd.DataFrame(boundary_weighted)
df3.to_csv("PA_GOPBW_50000_20190721")
