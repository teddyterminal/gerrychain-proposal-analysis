from gerrychain.random import random
from gerrychain.tree import *

def recom(partition, pop_col, pop_target, epsilon, node_repeats):
    """ReCom proposal.

    Description from MGGG's 2018 Virginia House of Delegates report:
    At each step, we (uniformly) randomly select a pair of adjacent districts and
    merge all of their blocks in to a single unit. Then, we generate a spanning tree
    for the blocks of the merged unit with the Kruskal/Karger algorithm. Finally,
    we cut an edge of the tree at random, checking that this separates the region
    into two new districts that are population balanced.

    Example usage::

        from functools import partial
        from gerrychain import MarkovChain
        from gerrychain.proposals import recom

        # ...define constraints, accept, partition, total_steps here...

        # Ideal population:
        pop_target = sum(partition["population"].values()) / len(partition)

        proposal = partial(
            recom, pop_col="POP10", pop_target=pop_target, epsilon=.05, node_repeats=10
        )

        chain = MarkovChain(proposal, constraints, accept, partition, total_steps)

    """
    edge = random.choice(tuple(partition["cut_edges"]))
    parts_to_merge = (partition.assignment[edge[0]], partition.assignment[edge[1]])

    subgraph = partition.graph.subgraph(
        partition.parts[parts_to_merge[0]] | partition.parts[parts_to_merge[1]]
    )

    flips = recursive_tree_part(
        subgraph,
        parts_to_merge,
        pop_col=pop_col,
        pop_target=pop_target,
        epsilon=epsilon,
        node_repeats=node_repeats,
    )

    return partition.flip(flips)

def bipartition_tree(
    graph,
    pop_col,
    pop_target,
    epsilon,
    node_repeats,
    spanning_tree=None,
    choice=random.choice,
):
    """This function finds a balanced 2 partition of a graph by drawing a
    spanning tree and finding an edge to cut that leaves at most an epsilon
    imbalance between the populations of the parts. If a root fails, new roots
    are tried until node_repeats in which case a new tree is drawn.

    Builds up a connected subgraph with a connected complement whose population
    is ``epsilon * pop_target`` away from ``pop_target``.

    Returns a subset of nodes of ``graph`` (whose induced subgraph is connected).
    The other part of the partition is the complement of this subset.

    :param graph: The graph to partition
    :param pop_col: The node attribute holding the population of each node
    :param pop_target: The target population for the returned subset of nodes
    :param epsilon: The allowable deviation from  ``pop_target`` (as a percentage of
        ``pop_target``) for the subgraph's population
    :param node_repeats: A parameter for the algorithm: how many different choices
        of root to use before drawing a new spanning tree.
    :param spanning_tree: The spanning tree for the algorithm to use (used when the
        algorithm chooses a new root and for testing)
    :param choice: :func:`random.choice`. Can be substituted for testing.
    """
    populations = {node: graph.nodes[node][pop_col] for node in graph}

    balanced_subtree = None
    if spanning_tree is None:
        spanning_tree = random_spanning_tree(graph)
    restarts = 0
    while balanced_subtree is None:
        if restarts == node_repeats:
            return None
        h = PopulatedGraph(spanning_tree.copy(), populations, pop_target, epsilon)
        balanced_subtree = contract_leaves_until_balanced_or_none(h, choice=choice)
        restarts += 1

    return balanced_subtree


def recursive_tree_part(graph, parts, pop_target, pop_col, epsilon, node_repeats=None):
    """Uses :func:`~gerrychain.tree_methods.bipartition_tree` recursively to partition a tree into
    ``len(parts)`` parts of population ``pop_target`` (within ``epsilon``). Can be used to
    generate initial seed plans or to implement ReCom-like "merge walk" proposals.

    :param graph: The graph
    :param parts: Iterable of part labels (like ``[0,1,2]`` or ``range(4)``
    :param pop_target: Target population for each part of the partition
    :param pop_col: Node attribute key holding population data
    :param epsilon: How far (as a percentage of ``pop_target``) from ``pop_target`` the parts
        of the partition can be
    :param node_repeats: Parameter for :func:`~gerrychain.tree_methods.bipartition_tree` to use.
    :return: New assignments for the nodes of ``graph``.
    :rtype: dict
    """
    flips = {}
    remaining_nodes = set(graph.nodes)

    for part in parts[:-1]:
        nodes = bipartition_tree(
            graph.subgraph(remaining_nodes),
            pop_col=pop_col,
            pop_target=pop_target,
            epsilon=epsilon,
            node_repeats=node_repeats,
        )

        if nodes == None: #we failed to recombine
            return {}


        for node in nodes:
            flips[node] = part
        # update pop_target?

        remaining_nodes -= nodes

    # All of the remaining nodes go in the last part
    for node in remaining_nodes:
        flips[node] = parts[-1]

    return flips

