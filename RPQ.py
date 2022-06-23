from parse import Lexer, Parser, Token, State, NFA, Handler, HandlerTree
from random import randrange
import numpy as np
from functools import reduce
import os
import re

bio_graph = "alibaba.graph.txt"
samplegraph = "papergraph.txt"
sampleq = "testqueriespaper.txt"
sampleMS = "testMSqueries.txt"
randomqueries = "samplerandomqueries.txt"
bio_queries = "bio_queries.txt"
bio_queries_1S = "bio_queries_single_src.txt"


# from regex github project
def compile(p, debug=False):
    def print_tokens(tokens):
        for t in tokens:
            print(t)

    lexer = Lexer(p)
    parser = Parser(lexer)
    tokens = parser.parse()

    handler = Handler()

    if debug:
        print_tokens(tokens)

    nfa_stack = []

    for t in tokens:
        handler.handlers[t.name](t, nfa_stack)

    assert len(nfa_stack) == 1
    return nfa_stack.pop()


def makeParseTree(p, debug=False):
    def print_tokens(tokens):
        for t in tokens:
            print(t)

    lexer = Lexer(p)
    parser = Parser(lexer)
    tokens = parser.parse()

    handler = HandlerTree()

    if debug:
        print_tokens(tokens)

    nfa_stack = []

    for t in tokens:
        handler.handlers[t.name](t, nfa_stack)

    assert len(nfa_stack) == 1
    return nfa_stack.pop()


def loadgraph(gfname):
    '''
    load graph from file
    input: file where each line has an edge as: node1 node2 label
    output: graph data structure: dict{ node:[(node2,label), (node3,label)...], node2:[] ...}
    '''
    grafile = open(gfname)
    thegraph = dict()
    cnt = 0
    for line in grafile:
        cnt += 1
        if (cnt % 10000 == 0): print(cnt)
        if (len(line) <= 1): continue
        tup = line.split()
        node1, node2, label = tup[0], tup[1], tup[2]
        thegraph.setdefault(node1, []).append((node2, label))
        thegraph.setdefault(node2, [])
    grafile.close()
    return thegraph

def loadgraphNT(gfname):
    '''
    load graph from file (n triples format)
    input: file where each line has an edge as: node1 label node2 .   
    nodes can be URI enclosed in <>, or literals
    output: graph data structure: dict{ node:[(node2,label), (node3,label)...], node2:[] ...}
    '''
    grafile = open(gfname)
    thegraph = dict()
    cnt = 0
    for line in grafile:
        cnt += 1
        #if (cnt % 10000 == 0): print(cnt)
        if (len(line) <= 1): continue
        tup = line.split()
        node1, label, node2 = tup[0][1:-1], tup[1][1:-1], (tup[2][1:-1] if tup[2].startswith('<http') else tup[2])
        thegraph.setdefault(node1, []).append((node2, label))
        thegraph.setdefault(node2, [])
    grafile.close()
    return thegraph


def loadgraph_w_rev(gfname):
    '''
    load graph from file
    input: file where each line has an edge as: node1 node2 label
    output: graph data structure: dict{ node:[(node2,label), (node3,label)...], node2:[] ...}
    + reverse edges = edges with label ^label
    '''
    grafile = open(gfname)
    thegraph = dict()
    cnt = 0
    for line in grafile:
        cnt += 1
        if (cnt % 10000 == 0): print(cnt)
        if (len(line) <= 1): continue
        tup = line.split()
        node1, node2, label = tup[0], tup[1], tup[2]
        reverse = "^"+label
        thegraph.setdefault(node1, []).append((node2, label))
        thegraph.setdefault(node2, []).append((node1, reverse))
    grafile.close()
    return thegraph


def add_reverse_links(graph):
    #print ("adding reverse links - graph size=", len(graph))
    for node1 in graph:
        for node2,edge in graph[node1]:
            if (edge.startswith("^")):
                continue
            reverse = "^"+edge
            graph.setdefault(node2, []).append((node1, reverse))

def loadgraphTxt(edgeList):
    '''
    load graph from a list of edges node1, edgelabel, node2
    input: list of tuples (node1, label, node2)
    output: graph data structure: dict{ node:[(node2,label), (node3,label)...], node2:[] ...}
    '''
    thegraph = dict()
    cnt = 0
    for tup in edgeList:
        cnt += 1
        if (cnt % 10000 == 0): print(cnt)
        node1, label, node2 = tup[0], tup[1], tup[2]
        thegraph.setdefault(node1, []).append((node2, label))
        thegraph.setdefault(node2, [])
    return thegraph


def inoutdegrees(g):
    ingraph = dict()
    for node in g.keys():
        for (othernode, label) in g[node]:
            ingraph.setdefault(othernode, []).append((node, label))
            ingraph.setdefault(node, [])
    inout = []
    for node in g.keys():
        outdeg = len(g[node])
        indeg = len(ingraph[node])
        inout.append((indeg, outdeg))

    return inout


def reducerfun(t1, t2):  # reducer function
    # expects two tuples of length 3, merges them if the 3rd element is the same,
    if (t1[-1][2] == t2[2]):  # first element is a list of 3-element tuples, second is a 3-element tuple
        middle = t1[-1][1]  # set of values in middle for given value in 3rd
        middle.add(t2[1])  # add value from second tuple
        return t1[:-1] + [(t1[-1][0], middle, t1[-1][2])]
    else:
        return t1 + [(t2[0], set(t2[1]), t2[2])]


def bfs(graph, NFA, start):
    '''
    This is the main algorithm where the product automaton is constructed and searched on the fly.
    '''
    visited, queue = set(), [(start, NFA.start)]
    edgelist = []  # list of traversed edges
    graphsolutions = set()
    broadcasts = set()

    while queue:
        vertex = queue.pop(0)
        if vertex not in visited:
            visited.add(vertex)
            vgraph, vautom = vertex
            # this step to be modified to follow specific edge labels

            # if vertex is terminal, add to solution list
            if (vautom.is_end):
                graphsolutions.add(vgraph)

            # get epsilon-transitions
            eps_states = [(vgraph, veps) for veps in vautom.epsilon]
            trans_states = []
            # get labeled transitions
            # record them as candidate broadcasts
            bccandidates = [(vgraph, lbl, vautom.transitions[lbl].name) for lbl in vautom.transitions.keys()]
            if (len(bccandidates) > 0):
                bccandidates.sort(key=lambda tup: tup[2])  # sort by destination
                initial = bccandidates.pop(0)  # initial value is removed from list
                reducedbc = reduce(reducerfun, bccandidates, [
                    (initial[0], set([initial[1]]), initial[2])])  # build reducing initializer from initial tuple
                # reducedbc is a list of tuples (fromnode, set(labels), to automstate)
                reducedhashable = [(tup[0], "|".join(sorted(list(tup[1])))) for tup in reducedbc]
                # now list of tuples (fromnode, "l1|l2|l3...|ln") -> to automstate left out
                broadcasts.update(reducedhashable)

            # for each neighbouring node in graph
            for (vg2, outlabel) in graph[vgraph]:
                if (outlabel in vautom.transitions):
                    vautom2 = vautom.transitions[outlabel]
                    trans_states.append((vg2, vautom2))
                    edgelist.append((vgraph, outlabel, vg2))

            trans_states.extend(eps_states)

            queue.extend([s for s in trans_states if s not in visited])

    return graphsolutions, visited, list(
        set(edgelist)), broadcasts  # set of graph nodes in terminal nodes of product automaton; list of visited nodes; list of traversed edges; list of broadcast queries


def bfsEstimator(probabilities, graphsize, NFA):
    '''
    This is the main algorithm where the product automaton is constructed and searched on the fly.
    In this one we use a random artificial graph to estimate the selectivity of a query.
    Probabilities is a dict {edge label L: probability that two nodes are connected with an edge labeled L}
    graphsize is the size of the graph.
    '''
    visited, queue = set(), [(0, NFA.start)]  # the start node is zero, which means some arbitrary node
    generatedgraph = dict()  # keep track of the graph as we've generated it
    gengraphlabels = dict()  # for each node, keep track of which labels we've generated for each node
    # dict {node:[label1, label2...]}
    graphsolutions = set()
    broadcasts = set()
    edgelist = []

    while queue:
        if (len(queue) > 100000) or len(visited) > 100000:
            print("queue length:" + len(queue) + len(visited))
            raise Exception
        vertex = queue.pop(0)
        if vertex not in visited:
            visited.add(vertex)
            vgraph, vautom = vertex

            # if vertex is terminal, add to solution list
            if (vautom.is_end):
                graphsolutions.add(vgraph)

            # get epsilon-transitions
            eps_states = [(vgraph, veps) for veps in vautom.epsilon]
            trans_states = []

            # get labeled transitions
            # record them as candidate broadcasts
            bccandidates = [(vgraph, lbl, vautom.transitions[lbl].name) for lbl in vautom.transitions.keys()]
            if (len(bccandidates) > 0):
                bccandidates.sort(key=lambda tup: tup[2])  # sort by destination
                initial = bccandidates.pop(0)  # initial value is removed from list
                reducedbc = reduce(reducerfun, bccandidates, [
                    (initial[0], set([initial[1]]), initial[2])])  # build reducing initializer from initial tuple
                # reducedbc is a list of tuples (fromnode, set(labels), to automstate)
                reducedhashable = [(tup[0], "|".join(sorted(list(tup[1])))) for tup in reducedbc]
                # now list of tuples (fromnode, "l1|l2|l3...|ln") -> to automstate left out
                broadcasts.update(reducedhashable)

            # randomly generate outgoing nodes from here, only for edges of interest
            # TODO: should we actually keep track of all generated edges?
            for label in (vautom.transitions.keys()):  # these are the outgoing labels we're interested in
                # have we generated this yet?
                if (vgraph not in gengraphlabels) or (label not in gengraphlabels[vgraph]):  # no!
                    # we're doing it now, so remember it
                    gengraphlabels.setdefault(vgraph, []).append(label)
                    # how many outgoing edges with this label from this node? binomial random variable
                    howmany = np.random.binomial(graphsize, probabilities[label])  # binomial (n,p)
                    if (howmany > 0):
                        print("generating " + str(howmany) + " edges!!")
                        # this seems really slow
                        # whichones = np.random.permutation(graphsize)[:howmany] #choose those outgoing edges without replacement
                        whichones = set()
                        while len(whichones) < howmany:
                            whichones.add(np.random.randint(graphsize))

                        generatedgraph.setdefault(vgraph, []).extend([(V, label) for V in whichones])
                # else: yes we've generated it and can just use it from before
            # now we've got randomly generated outgoing arcs from our node of interest, labeled with labels of interest!
            # now follow them in the standard way!
            # for each neighbouring node in graph

            for (vg2, outlabel) in generatedgraph.get(vgraph,
                                                      []):  # in case the above generated nothing (a probabilities issue)
                print("vg2! " + vg2 + " " + outlabel)
                if (outlabel in vautom.transitions):
                    vautom2 = vautom.transitions[outlabel]
                    trans_states.append((vg2, vautom2))
                    edgelist.append((vgraph, outlabel, vg2))

            trans_states.extend(eps_states)

            queue.extend([s for s in trans_states if s not in visited])

    return graphsolutions, visited, list(set(edgelist)), broadcasts


def bfsEstimatorBigram(probabilities, graphsize, NFA):
    '''
    same as above, except probabilities are given as `bigrams', probabilities of edges based on previous edge crossed
    '''
    visited, queue = set(), [(0, NFA.start, 0, 'INIT')]  # the start node is zero, which means some arbitrary node
    generatedgraph = dict()  # keep track of the graph as we've generated it
    gengraphlabels = dict()  # for each node, keep track of which labels we've generated for each node
    # dict {node:[label1, label2...]}
    graphsolutions = set()
    broadcasts = set()
    edgelist = []

    while queue:
        # if (len(queue)>100000) or len(visited)>100000:
        #    print "queue length:", len(queue), len(visited)
        #    raise Exception
        vertex = queue.pop(0)
        if (vertex[0], vertex[1]) not in visited:
            visited.add((vertex[0], vertex[1]))
            vgraph, vautom, prevnode, prevlabel = vertex

            # if vertex is terminal, add to solution list
            if (vautom.is_end):
                graphsolutions.add(vgraph)

            # get epsilon-transitions
            eps_states = [(vgraph, veps, prevnode, prevlabel) for veps in vautom.epsilon]
            trans_states = []

            # get labeled transitions
            # temporarily removing BC computation
            # record them as candidate broadcasts
            #            bccandidates = [(vgraph, lbl, vautom.transitions[lbl].name) for lbl in vautom.transitions.keys()]
            #            if (len(bccandidates)>0):
            #                bccandidates.sort(key=lambda tup: tup[2]) #sort by destination
            #                initial = bccandidates.pop(0)#initial value is removed from list
            #                reducedbc = reduce(reducerfun, bccandidates, [(initial[0], set([initial[1]]),initial[2])]) #build reducing initializer from initial tuple
            # reducedbc is a list of tuples (fromnode, set(labels), to automstate)
            #                reducedhashable = [(tup[0], "|".join(sorted(list(tup[1])))) for tup in reducedbc]
            #                #now list of tuples (fromnode, "l1|l2|l3...|ln") -> to automstate left out
            #                broadcasts.update(reducedhashable)

            # randomly generate outgoing nodes from here, only for edges of interest
            # TODO: should we actually keep track of all generated edges?
            for label in (vautom.transitions.keys()):  # these are the outgoing labels we're interested in
                # have we generated this yet?
                if (label not in gengraphlabels.get(vgraph, [])):  # no!
                    # we're doing it now, so remember it
                    gengraphlabels.setdefault(vgraph, []).append(label)
                    # how many outgoing edges with this label from this node? binomial random variable

                    whichones = set()
                    if (prevlabel == label):
                        howmany = np.random.binomial(graphsize - 1, probabilities[prevlabel].get(label,
                                                                                                 0)) + 1  # binomial (n-1,p)+1 because we arrived with one edge, so we know at least one exists.
                        whichones.add(
                            prevnode)  # there's  link back to the node we came from; this one is known
                    else:
                        howmany = np.random.binomial(graphsize, probabilities[prevlabel].get(label,
                                                                                             0))  # binomial (n,p)
                    while len(whichones) < howmany:
                        whichones.add(np.random.randint(graphsize))

                    generatedgraph.setdefault(vgraph, []).extend([(V, label) for V in whichones])
                # else: yes we've generated it and can just use it from before
            # now we've got randomly generated outgoing arcs from our node of interest, labeled with labels of interest!
            # now follow them in the standard way!
            # for each neighbouring node in graph

            for (vg2, outlabel) in generatedgraph.get(vgraph,
                                                      []):  # in case the above generated nothing (a probabilities issue)
                if (vautom.transitions.has_key(outlabel)):
                    vautom2 = vautom.transitions[outlabel]
                    trans_states.append((vg2, vautom2, vgraph, outlabel))
                    edgelist.append((vgraph, outlabel, vg2))

            trans_states.extend(eps_states)

            queue.extend([s for s in trans_states if (s[0], s[1]) not in visited])

    return graphsolutions, visited, list(set(edgelist)), broadcasts


def getLabelFrequencies(gfilename, gsize=None):
    '''
    get graph label frequencies from a file, as binomial probabilities that an edge with the given label exists between two arbitrary nodes.
    inputs:
    gfilename: a file (name) containing the list of graph edges, each line being node1 node2 label
    gsize: size of graph (number of nodes): this can be provided if some nodes don't appear in the file. Otherwise the number of nodes will be counted as the number of nodes that appear in the file.
    '''
    nodes = set()
    edgecount = dict()  # dict {label:count}
    gfile = open(gfilename)
    for line in gfile:
        vals = line.split()
        if (not gsize):  # only count nodes if if gsize isn't given
            nodes.add(vals[0])
            nodes.add(vals[1])
        edgecount[vals[2]] = edgecount.get(vals[2], 0) + 1
    gfile.close()

    if (not gsize):  # if the number of nodes isn't provided
        gsize = len(nodes)
    numnodes_squared = gsize * gsize

    for k in edgecount.keys():
        edgecount[k] = edgecount[k] * 1.0 / numnodes_squared  # convert to binomial probability, based on nodeset
    # note: this
    return edgecount, gsize  # in case gsize was not provided


def build2GramGraphModel(gfilename, gsize=None):
    '''
    get graph label frequencies from a file, as binomial probabilities that an edge with the given label exists between two arbitrary nodes, and conditional to preceding labels (first order markov model)
    inputs:
    gfilename: a file (name) containing the list of graph edges, each line being node1 node2 label
    gsize: size of graph (number of nodes): this can be provided if some nodes don't appear in the file. Otherwise the number of nodes will be counted as the number of nodes that appear in the file.
    '''
    g = loadgraph(gfilename)
    if (not gsize):
        gsize = len(g)
    counts1 = dict()
    len2seqs = dict()
    for node in g.keys():
        for (othernode, label) in g[node]:
            counts1[label] = counts1.get(label, 0) + 1  # increment count for unigrams
            for (thirdnode, label2) in g[othernode]:
                oldcount = len2seqs.setdefault(label, dict()).setdefault(label2, 0)
                len2seqs[label][label2] = oldcount + 1
    print("bigrams computed!")

    for l1 in len2seqs.keys():
        denom = counts1[l1] * gsize
        for l2 in len2seqs[l1].keys():
            if (
                    l1 == l2):  # special case because there is always one edge back to the previous node, we want to count the probabilities of the others. (checking for neg values with 0, this happens in a sample)
                len2seqs[l1][l2] = max(0, (len2seqs[l1][l2] - counts1[l1]) * 1.0 / (counts1[l1] * (gsize - 1)))
            else:
                len2seqs[l1][l2] = len2seqs[l1][l2] * 1.0 / denom
    # add in an extra set of edge probabilities for the initial step
    for lbl in counts1.keys():
        counts1[lbl] = counts1[lbl] * 1.0 / (gsize * gsize)
    len2seqs['INIT'] = counts1

    print("bigrams counts normalized!")
    return len2seqs, gsize


def runquery(graph, startnode, regex):
    '''
    Run provided single-source query on provided graph; query =(startnode,regex)
    returns answers + set of visited nodes in the graph
    '''
    NFA = compile(regex)
    #print("starting...")
    return bfs(graph, NFA, startnode)


def runMSquery(graph, regex):  # note: out of date doesn't handle broadcasts and edgelist
    '''
    run provided query (regex only) on provided graph
    returns answers + set of visited nodes in the graph
    '''
    NFA = compile(regex)
    print("starting...")
    answers = []
    visited = []
    edges = []
    broadcasts = []
    for startnode in graph:
        sol, vis, el, bc = bfs(graph, NFA, startnode)
        # for s in sol:
        #    answers.append((startnode,s))
        answers.append(len(sol))
        # if (len(vis)>1): #we necessarily visit the starting node, but let's say that doesn't count if we don't visit any other nodes from there
        # we must make this distinction because with our naive approach we necessarily visit the full graph.
        visited.append(len(set([v1 for (v1, v2) in vis])))  # how many graph nodes visited in the PAA?
        edges.append(len(el))  # how many graph edges visited in the PAA?)
        # each broadcast comes as=(fromnode, "l1|l2|l3...|ln")
        bccost = 0
        for msg in bc:
            bccost += len(msg[1].split('|')) + 1
        broadcasts.append(bccost)
    return answers, visited, edges, broadcasts


def __main__():
    pass


def singlesource(gfile, qfile, outfile=None):
    '''
    run single-source queries from provided file on provided graph
    optionally output results to a third file
    '''
    # load the graph
    g = loadgraph(gfile)

    print("loaded graph")
    qf = open(qfile)
    if (outfile):
        outf = open(outfile, 'w', 4096)
    cnt = 0
    for line in qf:
        cnt += 1
        # if(cnt%3):break
        snode, regex = line.split()[0], line.split()[1]
        if (snode not in g.keys()):
            print("[query start node node not in graph]")
            continue
        print(str(cnt) + ": query:\n" + snode + regex)
        if (outfile):
            outf.write("query:\n" + snode + ", " + regex + "\n")
        sol, vis, edgelist, bc = runquery(g, snode, regex)  # get: solutions, visited nodes, visited edges, broadcasts
        vnodes = set([v1 for (v1, v2) in vis])
        bc = sorted(['(' + v + ',' + a + ')' for (v, a) in bc])  # list+stringify set of broadcasts
        el = sorted(['(' + v + ',' + a + ',' + v2 + ')' for (v, a, v2) in edgelist])  # list+stringify set of broadcasts
        if (outfile):
            outf.write("solution_nodes:\n")
            outf.write(" ".join(sol) + "\n")
            outf.write("visited_nodes:\n")
            outf.write(" ".join(vnodes) + "\n")
            outf.write("broadcasts:\n")
            outf.write(" ".join(bc) + "\n")
            outf.write("edgelist:\n")
            outf.write(" ".join(el) + "\n")
        else:
            print("solutions:\n" + sol)
            print("visitednodes:\n")
            if (len(vnodes) > 30):
                print("(" + str(len(vnodes)) + ")")
            else:
                print(vnodes)
            print("broadcasts:\n")
            if (len(bc) > 30):
                print("(" + str(len(bc)) + ")")
            else:
                print(bc)
    if (outfile):
        outf.close()


def multisourceEstimation(gfile, qfile, gsize=None, outfile=None, reps=10):
    '''
    run estimator for single-source queries using bfsEstimator
    inputs:
    -gfile, a file to sample estimates from
    -gsize, a given graph size (nodes) to use in simulations (optional, if not given use the number of nodes in gfile)
    optionally output results to a third file
    '''
    #
    # probabilities, gsize = getLabelFrequencies(gfile, gsize) #swtich between these lines to go for bigram of unigram-based estimates
    probabilities, gsize = build2GramGraphModel(gfile, gsize)
    print("probabilities:" + str(probabilities))
    qf = open(qfile)
    if (outfile):
        outf = open(outfile, 'w', 4096)

    nfas = []
    for line in qf:
        # if(cnt%3):break
        regex = line.strip()  # here we use the multi-source query def: only regex
        NFA = compile(
            regex)  # only compile the regex once... not that it's super costly, but let's not do it thousands of times!
        nfas.append(NFA)

    for repeat in range(reps):  # repeat k times the full list of experiments.
        cnt = 0

        for NFA in nfas:  # just iterate over the NFAS directly
            cnt += 1
            if (repeat % 500 == 0): print(str(cnt) + str(repeat))
            if (outfile):
                outf.write("q" + str(cnt) + "\t")
                # --------run the query --------------
            # sol, vis, edgelist, bc = bfsEstimator(probabilities, gsize, NFA) #get: solutions, visited nodes, visited edges, broadcasts
            sol, vis, edgelist, bc = bfsEstimatorBigram(probabilities, gsize,
                                                        NFA)  # get: solutions, visited nodes, visited edges, broadcasts
            # --------write results to file ----------
            vnodes = set([v1 for (v1, v2) in vis])
            bc = sorted(['(' + str(v) + ',' + a + ')' for (v, a) in bc])  # list+stringify set of broadcasts
            totalcost = sum([len(re.findall('[1-9]+', message)) for message in
                             bc])  # for each broadcast message, get all edges and nodes (numbers) and count them. Then sum over all messages.

            # =============this is the data we'll have in the file!
            data = [len(sol), len(vnodes), len(bc), totalcost, len(edgelist)]
            # =================================================
            if (outfile):
                outf.write("\t".join([str(d) for d in data]) + "\n")

                # print data

        # =================================================

    if (outfile):
        outf.close()


def multisource(gfile, qfile, outfile=None):
    g = loadgraph(gfile)
    print("loaded graph")
    qf = open(qfile)
    if (outfile):
        outf = open(outfile, 'w', 4096)
        # cnt=0
    for line in qf:
        # cnt =+1
        # if(cnt>3):break
        regex = line.strip()
        print("multi-source query:" + regex)
        if (outfile):
            outf.write("query:\n" + regex + "\n")
        sols, vnodes = runMSquery(g, regex)
        if (outfile):
            outf.write("solution_pairs:\n")
            outf.write(" ".join(['(' + v1 + ',' + v2 + ')' for (v1, v2) in sols]) + "\n")
            outf.write("visited_nodes:\n")
            outf.write(" ".join(vnodes) + "\n")
        else:
            print("solutions:\n" + str(sol))
            print("visitednodes\n:")
            if (len(vnodes) > 30):
                print("(" + str(len(vnodes)) + ")")
            else:
                print(vnodes)

    if (outfile):
        outf.close()


def multisource2(gfile, qfile, outfile=None):
    g = loadgraph(gfile)
    print("loaded graph")
    qf = open(qfile)
    if (outfile):
        outf = open(outfile, 'w', 4096)
    cnt = 0
    for line in qf:
        cnt += 1
        # if(cnt>3):break
        regex = line.strip()
        print("multi-source query:" + regex)
        sols, vnodes, elist, bc = runMSquery(g, regex)
        if (outfile):
            for i in range(len(sols)):
                outf.write(
                    "q" + str(cnt) + "\t" + str(sols[i]) + "\t" + str(vnodes[i]) + "\t" + str(elist[i]) + "\t" + str(
                        bc[i]) + "\n")

    if (outfile):
        outf.close()


def selectNodes(graph, elist):
    '''
    select all nodes from the given graph such that one of their outgoing edges is in the provided list
    graph format: dict node:[(node1,label1), (node2,label2), ...]
    #deprecated!
    '''
    matchingnodes = set()
    for k in graph.keys():
        outedges = [label for (node, label) in graph[k]]
        for e in outedges:
            if e in elist:
                matchingnodes.add(k)
                break
    return matchingnodes


def selectNodes2(graph, elist):
    '''
    select all nodes from the given graph such that one of their outgoing edges is in the provided list
    this version returns the total size of data assuming a up2p-like data model
    graph format: dict node:[(node1,label1), (node2,label2), ...]
    => used for getting selectivity in the UP2P/web data model
    '''
    matchingnodes = []
    for k in graph.keys():
        outedges = [label for (node, label) in graph[k]]
        for e in outedges:
            if e in elist:
                matchingnodes.append(len(graph[k]))
                break
    return matchingnodes


def selectEdges(graph, elist):
    '''
    select all EDGES from the given graph such that its label is in the provided list
    graph format: dict node:[(node1,label1), (node2,label2), ...]
    '''
    matchingedges = set()
    for k in graph.keys():
        matchingedges.update([(k, node, label) for (node, label) in graph[k] if label in elist])

    return matchingedges


def selectForS1(gfile, qfile, outfile=None):
    '''
    get a priori selectiveness of queries:
    load a graph, load a list of queries, compute S1 selectiveness for each query.
        #DEPRECATED: parsefilecomplete() below does this and more.
    '''
    g = loadgraph(gfile)
    print("loaded graph")
    qf = open(qfile)
    if (outfile):
        outf = open(outfile, 'w', 4096)
        ostr = 'count\ts1nodes\ts1u/w\ts1e\n'  # header line
        outf.write(ostr)
    cnt = 0

    for line in qf:
        # find all edges in the query
        cnt += 1
        alledges = re.findall('(?<=<).*?(?=>)', line)
        # go through the graph and find the nodes that we would retrieve to process the query
        reduced = list(set(alledges))
        # print "query:", line,
        # print "edge labels: ", reduced, '['+str(len(reduced))+'] distinct edge labels'
        nodeset = selectNodes2(g, reduced)
        edgeset = selectEdges(g, reduced)
        if (outfile):
            ostr = str(cnt) + '\t' + str(len(nodeset)) + '\t' + str(sum(nodeset)) + '\t' + str(len(edgeset)) + '\n'
            outf.write(ostr)
        else:
            print("apriori nodeset size: " + str(len(nodeset)) + str(sum(nodeset)))  # here it's the sum of sizes
            print("apriori edgeset size: " + str(len(edgeset)))
    qf.close()


def sampler(size, outfile):
    # get a random sample of approximately 'size' queries from the queries file, of size 10k
    qfile = open("query1_newsyntax.TXT")
    # size of file is 10k
    totalsize = 10000
    ofile = open(outfile, "w")
    for line in qfile:
        if randrange(totalsize) < size:
            ofile.write(line)
    ofile.close()
    qfile.close()


def getS4selectiveness(thefile):
    # script to get the selectiveness of S4 in the up2p/web data model
    # parse results of query execution, for each visited node get size of node (all outgoing edges) in up2p/web DM
    # DEPRECATED: parsefilecomplete() below does this and more.
    graph = loadgraph(bio_graph)
    with open(thefile) as resfile:
        mode = 0  # starting
        for line in resfile:
            if line.startswith("query"):
                mode = 1
                continue
            elif line.startswith("solution"):
                mode = 2
                continue
            elif line.startswith("visited"):
                mode = 3
                continue
            elif line.startswith("broadcasts"):
                mode = 4
                continue

            # ok here we're getting data
            if (mode == 1):
                print(line)

            elif (mode == 2):
                pass
                # if(singlesrc):
                # parse_1S_line(line)
                # else:
                #    parse_MS_line(line)

            elif (mode == 3):
                vis = line.split()
                # here is where I have the list of visited nodes
                up2psize = sum([len(graph[v]) for v in vis])
                print("visited:" + str(len(vis)))
                print("up2psize:" + str(up2psize))


            elif (mode == 4):
                bc = line.split()
                print("broadcasts" + str(len(bc)))


def parsefilecomplete(thefile, graphfile, outfilename):
    '''
    parses a list of single-src RPQ results, with for each query successive lines for query, solution, number of visited nodes, list of broadcasts, list of traversed edges.
    gets the selectiveness for S1 and S4 as well.
    fields in output file are:

    field = q# querysize labels s1nodes S1sumnodesize s1edsges #solutions nodesvisitedPAA s4w bc sumbc s4edges
    index    0   1        2       3       4              5         6         7             8   9   10     11
    *warning* not the order expected in other parsing functions!! check before using, if needed, reorder!
    '''

    graph = loadgraph(graphfile)  # get graph for estimating S1/S4 selectiveness in web DM
    cnt = 0
    outfile = open(outfilename, 'w', 4096)
    with open(thefile) as resfile:
        mode = 0  # starting
        for line in resfile:
            if line.startswith("query"):
                cnt += 1
                if (cnt % 100 == 0): print(cnt)
                mode = 1
                continue
            elif line.startswith("solution"):
                mode = 2
                continue
            elif line.startswith("visited"):
                mode = 3
                continue
            elif line.startswith("broadcasts"):
                mode = 4
                continue
            elif line.startswith("edgelist"):
                mode = 5
                continue

            # ok here we're getting data
            if (mode == 1):  # query
                # get S1 selectiveness:
                alledges = re.findall('(?<=<).*?(?=>)', line)
                # go through the graph and find the nodes that we would retrieve to process the query
                reduced = list(set(alledges))  # reduce to set
                nodeset = selectNodes2(graph, reduced)
                edgeset = selectEdges(graph, reduced)
                data = [cnt, len(alledges), len(set(alledges)), len(nodeset), sum(nodeset), len(edgeset)]
                outfile.write('\t'.join([str(d) for d in data]) + '\t')
            elif (mode == 2):
                sols = line.split()
                outfile.write(str(len(sols)) + "\t")

            elif (mode == 3):
                vis = line.split()
                up2psize = sum([len(graph[v]) for v in vis])  # number of outgoing edges from this node
                outfile.write(str(len(vis)) + '\t' + str(up2psize) + '\t')

            elif (mode == 4):
                bc = line.split()
                # print "broadcasts", len(bc)
                # calculate sum of broadcast length
                totalcost = sum([len(re.findall('[1-9]+', message)) for message in
                                 bc])  # for each broadcast message, get all edges and nodes (numbers) and count them. Then sum over all messages.
                # print "totalcost=", totalcost
                outfile.write(str(len(bc)) + "\t" + str(totalcost) + "\t")
            elif (mode == 5):
                el = line.split()
                # print "edges", len(el)
                outfile.write(str(len(el)) + '\n')
    outfile.close()


def samplethegraph(graph, ratio, strat, dm, fnamebase='samplegraph', numberof=1):
    '''
    create subsets of the graph, and based on these, let's find out if we can estimate the selectivity of a query.
    parameters:
    + size ratio: 0.1 for 10% of the graph, for example.
    + strategies to create subsets:
         - random ('r')
         - random walks ('rw')
    + data model ('rdf' or 'web')
    + fnamebase = a name
    + number= how many samples
    Note: both strategies can be applied with either the RDF DM or the Web DM.
    '''
    # load the graph, if needed
    # graph = loadgraph(bio_graph)

    # --random selection-
    if (strat == 'r'):
        if (dm == 'rdf'):
            for index in range(numberof):
                selected = []
                for v in graph.keys():
                    for e in graph[v]:
                        if (np.random.uniform() < ratio):
                            selected.append([v, e[1], e[0]])
                outfile = open(
                    fnamebase + "_" + "{0:03d}".format(int(ratio * 1000)) + "_" + strat + "_" + dm + "_" + str(index),
                    "w")
                for e in selected:
                    outfile.write('\t'.join(e) + '\n')  # make a line with the edge
                outfile.close()
        elif (dm == 'web'):
            for index in range(numberof):
                selected = []
                for v in graph.keys():
                    if (np.random.uniform() < ratio):
                        for e in graph[v]:
                            selected.append([v, e[1], e[0]])
                outfile = open(
                    fnamebase + "_" + "{0:03d}".format(int(ratio * 1000)) + "_" + strat + "_" + dm + "_" + str(index),
                    "w")
                for e in selected:
                    outfile.write('\t'.join(e) + '\n')  # make a line with the edge
                outfile.close()

    elif (strat == 'rw'):
        for index in range(numberof):
            keylist = list(graph.keys())
            np.random.shuffle(keylist)
            teleportindex = 0
            selected = set()
            currentnode = keylist[0]
            expectedsize = ratio * sum([len(graph[v]) for v in graph.keys()])
            while (len(selected) < expectedsize):
                rand = np.random.uniform()
                if (rand < 0.15 or len(graph[currentnode]) == 0):
                    teleportindex = np.random.randint(len(keylist))
                    currentnode = keylist[teleportindex]
                else:
                    selectedge = graph[currentnode][np.random.randint(len(graph[currentnode]))]
                    theedge = [currentnode, selectedge[1], selectedge[0]]
                    if (dm == 'web'):
                        selected.update(['\t'.join([currentnode, s[1], s[0]]) for s in graph[currentnode]])
                    else:
                        selected.add('\t'.join(theedge))  # make a string with the edge (same above)
                    currentnode = theedge[2]
            outfile = open(
                fnamebase + "_" + "{0:03d}".format(int(ratio * 1000)) + "_" + strat + "_" + dm + "_" + str(index), "w")
            for e in selected:
                outfile.write(e + '\n')
            outfile.close()
    else:
        print(strat + ": not a known strategy")
        return


def allsampling():
    '''
    sample the graph using hard coded info. Produces 240 smaller graphs to work with.
    '''
    graph = loadgraph(bio_graph)
    for ratio in [0.001, 0.0025, 0.01, 0.025, 0.1, 0.25]:
        for strat in ['r', 'rw']:
            for dm in ['web', 'rdf']:
                print(ratio, dm, strat)
                samplethegraph(graph, ratio, strat, dm, 'samplegraph', 10)


def runQueriesOnSamples():
    qfile = 'bio_queries_1S_562.txt'
    for fname in os.listdir("./sampling/"):
        if fname.startswith("samplegraph"):
            # example:samplegraph_1000_rw_rdf_0
            print(fname)
            info = fname.split('_')
            # relevant fields : 1=ratio*1000, 2=random/randomwalk 3=rdf/web
            multisource("./sampling/" + fname, qfile, './results/results_' + '_'.join(info[1:]))

# allsampling()
# sampleMSqueries = "sampleMSqueries.txt"
# multisource(bio_graph,bio_queries, "RPQ_bio_results.txt")
# singlesource(bio_graph,bio_queries_1S, "RPQ_bio_results_single.txt")
# singlesource(bio_graph,'newsinglesrc_bio_queries_big.txt', 'newBioQueryResults_super2.txt')
# parsefilecomplete('newBioQueryResults_super2.txt', 'fullbioQselectivity.dat')
# runQueriesOnSamples()

# multisourceEstimation(bio_graph,'bio_queries.txt', gsize=None, outfile='estimates_unigram_devnull.txt', reps=5)
# multisource2(bio_graph,'bio_queries.txt',outfile='multivisited2.txt')
# multisourceEstimation(samplegraph,sampleMS, gsize=None, outfile='estimatesample.txt', reps=50)

# multisourceEstimation('samplegraph_1000_2.txt','bio_query_q1.txt', gsize=52050, outfile='estimates_bigram_sample.txt', reps=10000)