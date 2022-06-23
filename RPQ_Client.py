from RPQ import runquery, compile, bfs, add_reverse_links
from parse import NFA
from jinja2 import Environment, FileSystemLoader
import json, re, time
from urllib import parse, request
from functools import reduce
from itertools import groupby
import asyncio
from RPQ_Server_Proxy import Server_proxy

class Client:
    def __init__(self, name):
        self.name = name
        self.knownServers = None
        #self.logfile = None

    def expand_re(self, regex):
        '''
        Uses a regular expression in a format like <a><b>*<c> and decomposes it
        :param regex: regular expression to decompose
        :return: nothing; the expanded query automaton is added as an attribute to the Client object.
        '''

        NFA1 = compile(regex)
        DFA = NFA1.to_DFA()
        DFA.renameStates()
        if (self.debug):
            DFA.uglyprint()
        decomp = DFA.decomposePaths()
        #print("===== decomposed REGEX ======")
        # for k in decomp:
        #     rule = decomp[k]
        #     rule[2].printIndented(2)
        #     print(rule[0].name, rule[1].name, str(rule[2]))
        #     print(rule[0].name, rule[1].name, rule[2].sparql_str())
        decomp2 = {k: (decomp[k][0], decomp[k][1], str(decomp[k][2])) for k in decomp}
        end_states = set([s for s in DFA.allReachableStates() if (s.is_end)])
        self.expanded_re = decomp2, DFA.start, end_states

    def getServerByDomain(self, domain):
        for s in self.knownServers.values():
            if (s.domain == domain):
                return s


    def get_URI_domain(self, uri):
        '''
         find 'domain' substring of URI among known domains (list of servers)
         '''
        for s in self.knownServers.values():
            if (uri.startswith(s.domain)):
                return s.domain
        return None #no known domain found: either non-URI or else URI of unknown domain


    def get_data_graph(self):
        '''
        Get all the responses data structure from all the servers and merge then into a single graph
        :return: a graph of data, with nodes from the distirbuted graph, and edge labels = regular expressions (extended automaton)
        '''

        serv_list = list(self.knownServers.values())
        # send the query to each server [the server object acts as proxy and builds the SPARQL query specifically for each server]
        # retrieve a list of responses for each server
        list_results = []#list(map(lambda s: s.get_local_response(self.expanded_re, self.start_node), serv_list))
        for server in serv_list:
            new_graph = server.get_local_response(self.expanded_re, self.start_node, self.end_node)
            edgecounts = []
            if(self.debug):
                for n in new_graph:
                    edgecounts.append(len(set(new_graph[n])))
                print("data graph from server", server.name, ":", len(new_graph), "nodes", sum(edgecounts), "edges")
            list_results.append(new_graph)


        def merge_reducer(d1, d2):
            keys = set(d1.keys()).union(set(d2.keys()))
            newd = dict()
            for k in keys:
                set1 = d1.get(k, set())
                set2 = d2.get(k, set())
                newd[k] = set1.union(set2)
            return newd#{k: d1.get(k, []) + d2.get(k, []) for k in keys} #merged dictionaries

        full_result = reduce(merge_reducer, list_results)

        # Format the data_responses merged into a graph using the loadgraph format
        new_graph = dict()
        edgecnt=0
        with open("data_graph.debug.txt", "w") as logfile:
            for key in full_result:
                for nodepair in full_result[key]:
                    node1, node2 = nodepair
                    label = key
                    
                    if((node2, label) not in new_graph.get(node1,[])):
                        new_graph.setdefault(node1, []).append((node2, label))
                        logfile.write("\t".join([node1, label, node2])+"\n")
                        edgecnt+=1
                    new_graph.setdefault(node2, [])
        edgecnt = 0
        for n in new_graph:
            edgecnt += len(set(new_graph[n]))
                    
        print("data graph:", len(new_graph), "nodes,", edgecnt, "edges")
        return new_graph

    def test_NFA(self, regex):
        self.expand_re(regex)
        decomp, qa_start_state, qa_end_states = self.expanded_re

        print("===== decomposed REGEX ======")
        for k in decomp:
            rule = decomp[k]
            print(rule[0].name, rule[1].name, rule[2][:50])
            #print(rule[0].name, rule[1].name, rule[2].sparql_str())
        print("start state",qa_start_state.name)
        print("end states",[q.name for q in qa_end_states])
        NFA = self.get_NFA()
        NFA.uglyprint()
        decomp, qa_start_state, qa_end_states = self.expanded_re

        print("===== decomposed REGEX (after) ======")
        for k in decomp:
            rule = decomp[k]
            print(rule[0].name, rule[1].name, rule[2][:50])
            #print(rule[0].name, rule[1].name, rule[2].sparql_str())
        print("start state",qa_start_state.name)
        print("end states",[q.name for q in qa_end_states])

    def get_NFA(self):
        '''
        Retrieves the generalized automaton as an NFA object (using the State and NFA classes from parse.py)
        ** the transitions are unique rule identifiers rather than regular expressions **
        :return: an NFA object representing the generalized automaton associated with the regex at hand
        '''

        list_of_states = []
        transitions = dict()
        rules = self.expanded_re[0]
        end_state = None

        
        # Creates a list of all the states from the expanded_re rules
        for rule in rules:
            for state in rules[rule][:2]:
                if state not in list_of_states:
                    list_of_states.append(state)

        # Get all the transitions for each State
        for rule in rules:
            state1, state2, label = rules[rule][0], rules[rule][1], rule 
            transitions.setdefault(state1, set())
            transitions[state1].add((state2, rule))

        # Transform into a set of transition in dict and add them to the actual state
        for state in list_of_states:
            actual_transitions = {}
            for otherstate,rule in transitions.get(state,[]): #transition
                actual_transitions[rule]= otherstate   #actual_transitions[rule]=target state #.setdefault(transition[1], transition[0])
            state.transitions = actual_transitions

        # Get first state


        start_state = self.expanded_re[1]

        # list_of_states.remove(start_state)

        # # Get last states
        # for state in list_of_states:
        #     if state.is_end:
        #         end_state = state
        #         list_of_states.remove(state)

        # Set start and end states

        result_NFA = NFA(start_state)

        # Add all the states in between
        #for inter_state in list_of_states:
        #    result_NFA.addstate(inter_state, set())

        return result_NFA

    def set_servers_in_out_nodes(self, regex):
        '''
        Sets the all the outnodes and innodes for all the Knownservers of the Client instance with the given responses
        :param list_of_servers: List of Serveur instances for the given Client
        :return: None
        '''
        # Add all outnodes for each client.knownservers
        all_out_nodes = set()
        serv_list = list(self.knownServers.values())

        all_out_nodes = reduce(lambda a,b: a.union(b), map(lambda s: s.get_outgoing_nodes(regex), serv_list))
        
        all_innodes = {name: [] for name in self.knownServers}

        #assign each node to its server/domain
        for node in all_out_nodes:
            for name in all_innodes:
                if (node.startswith(self.knownServers[name].domain)):
                    all_innodes[name].append(node)

        if(self.debug):
            print("------ incoming nodes --", all_innodes)
        # tell each server which are its innodes
        for name in self.knownServers:
            #TODO: temp solution: add start_node as incoming node to its server
            #if(self.start_node.startswith(self.knownServers[name].domain)):
            #    all_innodes[name].append(self.start_node)
            
            self.knownServers[name].set_innodes(all_innodes[name])



    def initiate(self, list_of_servers, debugging=False):
        '''
        Set up list of servers to be used for distirbuted queries. 
        :param list_of_servers: sets the list of servers in the Client object
        :param debugging: indicates whether we debug the querying process on standard out (debug mode)
        :return:
        '''
        self.knownServers = {s.name: s for s in list_of_servers} #map each server name to the server object
        self.debug = debugging
        tt= time.time()
        for s in list_of_servers:
            s.set_debugging(debugging, tt)
        #self.start_time = tt
        

    def run_query_partial(self, regex, start_node=None, end_node=None, logfile=None, prefixes=None, one_query=True):
        '''
        Main method of the program
        Run a distributed query on the servers to collect partial responses, then run a local query on the resulting local graph
        :param regex: the regular expression in the query
        :param start_node: The start node for the path (single-source or boolean query)
        :param start_node: The end node for the path (single-source or boolean query)

        :param prefixes: a dict of prefixes to be used in the SPARQl query (to shorten URIs)
        :return:
        '''
        if(logfile):
            lf = open(logfile, "w")
        #reset time for logging
        tt= time.time()
        self.start_time = tt

        if (not self.knownServers):
            print("List of servers to query has not been set.\n Use method client.initiate([server1, server2...]) with Serveur objects.")
            return
        for s in self.knownServers.values():
            if(prefixes):
                s.set_query_prefixes(prefixes)
            if(not one_query):
                s.one_query=False
            if(logfile):
                #print("setting logfile!")
                s.set_logfile(lf) # sets all log files
            s.set_debugging(s.debug, tt)


        # ==== query info:
        self.start_node = start_node
        self.end_node = end_node
        self.expand_re(regex) #build generalized query automaton (see below)
        #query automaton is triple rule_dict, start node, end_states:
        #                       rule_dict:  keys are arbitrary names, values are triples state1, state2, regex
        #                                   state1, state2 refer to states of original automaton for original regex
        #                                   regex labels possible paths from state1 to state2    
        #                       rule_dict example format:
        #                       {"r1": (states[0], states[1], "<a><b>*"),
        #                        "r2": (states[0], states[2], "<a><b>*<c>"),
        #                        "r3": (states[0], states[3], "<a><b>*<c><d>"),
        #                        "r4": (states[1], states[1], "<b>+"),
        #                        "r5": (states[1], states[2], "<b>*<c>"),
        #                        "r6": (states[1], states[3], "<b>*<c><d>"),
        #                        "r7": (states[2], states[3], "<d>")}

        decomp, qa_start_state, qa_end_states = self.expanded_re

        #print("===== decomposed REGEX ======")
        for k in decomp:
            rule = decomp[k]
            #print(rule[0], rule[0].name, rule[1], rule[1].name, rule[2][:50])
            #print(rule[0].name, rule[1].name, rule[2].sparql_str())
        #print("start state",qa_start_state, qa_start_state.name)
        #print("end states",[(q, q.name) for q in qa_end_states])

        # Set all the in and out nodes [includes first SPARQL query in case of real servers]
        self.set_servers_in_out_nodes(regex) #go query-specific using regex
        #TODO: make it optional to use query-specific outnodes list

        # Get local graph with all the responses
        responses = self.get_data_graph()

        temp_NFA = self.get_NFA()
        if(self.debug):
            print("------ data graph: [generalized automaton]", responses)
            print("------ query generalized automaton")
            temp_NFA.uglyprint()
        # Create NFA with the expanded regex
        

        # Run query on the local graph with the expanded regex NFA
        if(self.start_node):
            if (self.start_node not in responses):# start node not in graph
                sol = set()
                print("zero paths from start node")
            else:
                sol, v, e, b = bfs(responses, temp_NFA, self.start_node) #function returns a 4-tuple, here we only care about 1st element
                if(self.end_node): #start AND end nodes => boolean query
                    return (self.end_node in sol) # => return True or False (shortcuts returning list of nodes or node pairs)
                else:
                    print("single-source RPQ:", len(sol), "answers") #sol will be returned below
            solutions = sol
        else: #multi-source query or single-source in reverse
            if self.end_node:   #single-source reverse
                print("reverse RPQ!")
                add_reverse_links(responses)
                nfar = temp_NFA.reversed()
                nfar.uglyprint()
                solutions, v, e, b = bfs(responses, nfar , self.end_node) #search back from end node
                print("single-source RPQ in reverse:", len(solutions), "answers") #sol will be returned below
            else:       
                snodes = list(responses.keys())
                solutions =[]
                for node1 in snodes:
                    sol, v, e, b = bfs(responses, temp_NFA, node1) #function returns a 4-tuple, here we only care about 1st element
                    solutions.extend([(node1, node2) for node2 in sol])

        if(logfile):
            lf.close()

        return solutions

    # def run_multi_source_query(self, regex, logfile=None, prefixes=None, one_query=True):
    #     '''
    #     Main method of the program
    #     Run a distributed query on the servers to collect partial responses, then run a local query on the resulting local graph
    #     :param regex: the regular expression in the query
    #     :param start_node: The start node for the single-source query
    #     :param prefixes: a dict of prefixes to be used in the SPARQl query (to shorten URIs)
    #     :return:
    #     '''
    #     if(logfile):
    #         lf = open(logfile, "w")
    #     #reset time for logging
    #     tt= time.time()
    #     self.start_time = tt

    #     if (not self.knownServers):
    #         print("List of servers to query has not been set.\n Use method client.initiate([server1, server2...]) with Serveur objects.")
    #         return
    #     for s in self.knownServers.values():
    #         if(prefixes):
    #             s.set_query_prefixes(prefixes)
    #         if(not one_query):
    #             s.one_query=False
    #         if(logfile):
    #             #print("setting logfile!")
    #             s.set_logfile(lf) # sets all log files
    #         s.set_debugging(s.debug, tt)


    #     # ==== single-source query info:
    #     self.start_node = start_node
    #     self.expand_re(regex) #build generalized query automaton (see below)
    #     #query automaton is triple rule_dict, start node, end_states:
    #     #                       rule_dict:  keys are arbitrary names, values are triples state1, state2, regex
    #     #                                   state1, state2 refer to states of original automaton for original regex
    #     #                                   regex labels possible paths from state1 to state2    
    #     #                       rule_dict example format:
    #     #                       {"r1": (states[0], states[1], "<a><b>*"),
    #     #                        "r2": (states[0], states[2], "<a><b>*<c>"),
    #     #                        "r3": (states[0], states[3], "<a><b>*<c><d>"),
    #     #                        "r4": (states[1], states[1], "<b>+"),
    #     #                        "r5": (states[1], states[2], "<b>*<c>"),
    #     #                        "r6": (states[1], states[3], "<b>*<c><d>"),
    #     #                        "r7": (states[2], states[3], "<d>")}

    #     decomp, qa_start_state, qa_end_states = self.expanded_re

    #     #print("===== decomposed REGEX ======")
    #     for k in decomp:
    #         rule = decomp[k]
    #         #print(rule[0], rule[0].name, rule[1], rule[1].name, rule[2][:50])
    #         #print(rule[0].name, rule[1].name, rule[2].sparql_str())
    #     #print("start state",qa_start_state, qa_start_state.name)
    #     #print("end states",[(q, q.name) for q in qa_end_states])

    #     # Set all the in and out nodes [includes first SPARQL query in case of real servers]
    #     self.set_servers_in_out_nodes(regex) #go query-specific using regex
    #     #TODO: make it optional to use query-specific outnodes list

    #     # Get local graph with all the responses
    #     responses = self.get_data_graph()

    #     temp_NFA = self.get_NFA()
    #     if(self.debug):
    #         print("------ data graph: [generalized automaton]", responses)
    #         print("------ query generalized automaton")
    #         temp_NFA.uglyprint()
    #     # Create NFA with the expanded regex
        

    #     # Run query on the local graph with the expanded regex NFA
    #     if (self.start_node not in responses):# start node not in graph
    #         sol = set()
    #         print("zero paths from start node")
    #     else:
    #         sol, v, e, b = bfs(responses, temp_NFA, self.start_node) #function returns a 4-tuple, here we only care about 1st element
    #     print("single-source RPQ:", len(sol), "answers")

    #     if(logfile):
    #         lf.close()

    #     return sol
    
    
    def iterative_strategy_1sq(self, start_node, regex, logfile=None, one_query=True, prefixes=None):

        '''
        Running the product automaton algorithm on the generalized query automaton, constructed and searched on the fly.
        '''

        if(logfile):
            lf = open(logfile, "w")

        #reset time for logging
        tt= time.time()
        self.start_time = tt


        if (not self.knownServers):
            print("List of servers to query has not been set.\n Use method client.initiate([server1, server2...]) with Serveur objects.")
            return
        for s in self.knownServers.values():
            if(prefixes):
                s.set_query_prefixes(prefixes)
            if(not one_query):
                s.one_query=False
            if(logfile):
                s.set_logfile(lf) # sets all log files
            s.set_debugging(s.debug, tt)


        self.start_node = start_node
        self.expand_re(regex) #build generalized query automaton
        rules, qa_start_state, qa_end_states = self.expanded_re
        decomp=rules
        #print("===== decomposed REGEX ======")
        for k in decomp:
            rule = decomp[k]
        #    print(rule[0], rule[0].name, rule[1].name, rule[2][:50])
            #print(rule[0].name, rule[1].name, rule[2].sparql_str())
        #print("start state",qa_start_state.name)
        #print("end states",[q.name for q in qa_end_states])
        def get_regex(id):
            return rules[id][2]
        def get_to_state(id):
            return rules[id][1]
        NFA = self.get_NFA() #query automaton, where transitions are identifiers from above 'rules'
        #print("------ query generalized automaton")
        #NFA.uglyprint()
        #print("qa end states:", [q.name for q in qa_end_states])

        visited = set()
        queue= [(self.start_node, NFA.start)]
        edgelist = []  # list of traversed edges
        graphsolutions = set()

        while queue:
            first_state = queue[0][1]
            next_query_vertices = set()
            next_queue = []
            for vp in queue: #vp = pair (graph node, autom state)
                if (vp in visited):
                    continue
                if (vp[1] == first_state):  #select all nodes related to the same automaton state
                    next_query_vertices.add(vp)
                    visited.add(vp)
                else:
                    next_queue.append(vp)
            queue = next_queue #enqueue the rest for next round            

            #---------- We have first_state = state of query automaton that we're exploring from
            #                   next_query_vertices = list of graph nodes that we want to connect from
            # we're exploring simultaneously a large number of vertices in product automaton, all the unexplored ones associated with a given automaton state
            
            # if vautom is end state of automaton, add graph nodes to solution list
            #if (first_state.is_end):
            #    graphsolutions.extend(next_query_vertices)
                    
            # (no epsilon-transitions in this situation: we have a deterministic but incomplete generalized automaton)
            assert(len(first_state.epsilon)==0)

            # prepare a graph query for each transition from the current automaton state
            nql = sorted(next_query_vertices,key=lambda p: self.get_URI_domain(p[0])) #sort necessary for groupby: sorting toseparate the relevant nodes for each server
            for domain, nodes in groupby(nql, lambda p: self.get_URI_domain(p[0])):
                server = self.getServerByDomain(domain)
                qn = [n for n,s in nodes]
                querymap = {}
                #select relevant subqueries
                for ident,to_state in first_state.transitions.items():
                    regex = get_regex(ident)
                    subquery = server.prepare_subquery(first_state, to_state, regex, qn, None, ident)
                    querymap[ident] = subquery
                sparql_queries = server.subqueries_to_sparql(querymap)
                #run queries!
                if(len(sparql_queries)==1): # one query
                    q = sparql_queries[0]
                    # execute query, get response as JSON
                    j = server.sparqlQuery(q)
                    dlist = Server_proxy.makedicts(j)
                else:                   # multiple queries
                    dlist=[]
                    for sq in sparql_queries:
                        ji = server.sparqlQuery(sq)
                        # merge all of these json results
                        dlist = dlist + Server_proxy.makedicts(ji)            
                    
                # follow the edges that we've just obtained
                reached_nodes = []
                for edge_id,subq in querymap.items():
                    #info for this rule:
                    node1, is_var1, node2, is_var2 = subq.get_vars()
                    
                    #remove ? marks from variable names
                    node1v = node1[1:] if(is_var1) else node1[1:-1]
                    node2v = node2[1:] if(is_var2) else node2[1:-1]
                    # get pairs for the considered edge
                    pairs = [(d.get(node1v, node1v), d.get(node2v, node2v)) for d in dlist if (node1v in d or node2v in d)]
                    reached_state = get_to_state(edge_id)
                    for pp in pairs:
                        reached_nodes.append((pp[1],reached_state))
                        edgelist.append((pp[0], edge_id, pp[1]))
                for vg, va in reached_nodes:
                    if (va in qa_end_states):
                        graphsolutions.add(vg)
                        #if (va != first_state or not(vg.startswith(domain))): #only continue exploring if we reached this node from a different server or a different state
                        if (not(vg.startswith(domain))): #only continue exploring if we reached this node from a different server or a different state
                            queue.append((vg,va))
                    else:
                        queue.append((vg,va)) # enqueue for next query
            
        if(logfile):
            lf.close()


        return graphsolutions, edgelist#)), broadcasts  # set of graph nodes in terminal nodes of product automaton; list of visited nodes; list of traversed edges; list of broadcast queries


    def test_sparql_queries(self, start_node, regex, prefixes=None, one_query=True, realstep1=False, query_specific=True):
        '''
        Create SPARQL queries for "dry run" test
        :param regex: the regular expression in the query
        :param start_node: The start node for the single-source query
        :param prefixes: a dict of prefixes to be used in the SPARQl query (to shorten URIs)
        :return:
        '''
        if (not self.knownServers):
            print("List of servers to query has not been set.\n Use method client.initiate([server1, server2...]) with Serveur objects.")
            return
        for s in self.knownServers.values():
            if(prefixes):
                s.set_query_prefixes(prefixes)
            if(not one_query):
                s.one_query=False

        # ==== single-source query info:
        self.start_node = start_node
        self.expand_re(regex) #build query automaton (see below)
        #query automaton is triple rule_dict, start node, end_states:
        #                       rule_dict:  keys are arbitrary names, values are triples state1, state2, regex
        #                                   state1, state2 refer to states of original automaton for original regex
        #                                   regex labels possible paths from state1 to state2    
        #                       rule_dict example format:
        #                       {"r1": (states[0], states[1], "<a><b>*"),
        #                        "r2": (states[0], states[2], "<a><b>*<c>"),
        #                        "r3": (states[0], states[3], "<a><b>*<c><d>"),
        #                        "r4": (states[1], states[1], "<b>+"),
        #                        "r5": (states[1], states[2], "<b>*<c>"),
        #                        "r6": (states[1], states[3], "<b>*<c><d>"),
        #                        "r7": (states[2], states[3], "<d>")}


        # Set all the in and out nodes [includes first SPARQL query in case of real servers]
        #self.set_servers_in_out_nodes(regex) #go query-specific using regex
        
        # for each server:
        # create step 1 query 
        # make fake list of incoming nodes for each server,
        # then create step 2 sparql queries with that fake data
        for sn,serv in self.knownServers.items():
            if(realstep1):
                if(query_specific):
                    self.set_servers_in_out_nodes(regex)
                else:
                    self.set_servers_in_out_nodes()
            else:
                if(query_specific):
                    q1 = serv.make_sparql_query_outgoing_nodes(regex)
                else:
                    q1 = serv.make_sparql_query_outgoing_nodes()
                print("Server", sn, "outgoing nodes query:=========================\n")
                print (q1)
        
                fakenodes = [serv.domain +"/alibaba#in"+str(i) for i in range(5)]
                serv.set_innodes(fakenodes)
            subq = serv.prepare_query2(self.expanded_re, self.start_node)
            qs = serv.subqueries_to_sparql(subq)

            print("Server", sn, "step2 query===========\n")
            for q in qs:
                print(q)

class AsyncClient(Client):
    def __init__(self, name):
        self.name = name
        self.knownServers = None

    def close_all_sessions(self):
        '''
         find 'domain' substring of URI among known domains (list of servers)
         '''
        loop = asyncio.get_event_loop()
        tasks = map(lambda s: asyncio.create_task(s.close_session()), self.knownServers.values())

        finished, unfinished = loop.run_until_complete(asyncio.wait(tasks)) #run all tasks asyncronously



    def get_data_graph(self):
        '''
        Get all the responses data structure from all the servers and merge then into a single graph
        :return: a graph of data, with nodes from the distirbuted graph, and edge labels = regular expressions (extended automaton)
        '''

        serv_list = list(self.knownServers.values())
        # send the query to each server [the server object acts as proxy and builds the SPARQL query specifically for each server]

        # asynchronously retrieve a list of responses for each server
        loop = asyncio.get_event_loop()
        tasks = map(lambda s: asyncio.create_task(s.get_local_response(self.expanded_re, self.start_node)), serv_list)
        finished, unfinished = loop.run_until_complete(asyncio.wait(tasks)) #run all tasks asyncronously
        list_results = map(lambda task: task.result(), finished)

        def merge_reducer(d1, d2):
            keys = set(d1.keys()).union(set(d2.keys()))
            newd = dict()
            for k in keys:
                set1 = d1.get(k, set())
                set2 = d2.get(k, set())
                newd[k] = set1.union(set2)
            return newd#{k: d1.get(k, []) + d2.get(k, []) for k in keys} #merged dictionaries

        full_result = reduce(merge_reducer, list_results)

        # Format the data_responses merged into a graph using the loadgraph format
        new_graph = dict()
        edgecnt=0
        with open("data_graph.debug.txt", "w") as logfile:
            for key in full_result:
                for nodepair in full_result[key]:
                    node1, node2 = nodepair
                    label = key
                    
                    if((node2, label) not in new_graph.get(node1,[])):
                        new_graph.setdefault(node1, []).append((node2, label))
                        logfile.write("\t".join([node1, label, node2])+"\n")
                        edgecnt+=1
                    new_graph.setdefault(node2, [])
        edgecnt = 0
        for n in new_graph:
            edgecnt += len(set(new_graph[n]))
                    
        print("data graph:", len(new_graph), "nodes,", edgecnt, "edges")
        return new_graph

    def set_servers_in_out_nodes(self, regex):
        '''
        Sets the all the outnodes and innodes for all the Knownservers of the Client instance with the given responses
        :param list_of_servers: List of Serveur instances for the given Client
        :return: None
        '''
        # Add all outnodes for each client.knownservers
        all_out_nodes = set()
        serv_list = list(self.knownServers.values())

        #get outnodes asynchronously
        loop = asyncio.get_event_loop()
        tasks = map(lambda s: asyncio.create_task(s.get_outgoing_nodes(regex)), serv_list)
        finished, unfinished = loop.run_until_complete(asyncio.wait(tasks))
        list_results = map(lambda task: task.result(), finished)

        all_out_nodes = reduce(lambda a,b: a.union(b), list_results)
        
        all_innodes = {name: [] for name in self.knownServers}

        #assign each node to its server/domain
        for node in all_out_nodes:
            for name in all_innodes:
                if (node.startswith(self.knownServers[name].domain)):
                    all_innodes[name].append(node)

        if(self.debug):
            print("------ incoming nodes --", all_innodes)
        # tell each server which are its innodes
        for name in self.knownServers:
            #TODO: temp solution: add start_node as incoming node to its server
            #if(self.start_node.startswith(self.knownServers[name].domain)):
            #    all_innodes[name].append(self.start_node)
            
            self.knownServers[name].set_innodes(all_innodes[name])


    async def runsubqueries_async(self, server, querymap): # run a query (set of subqueries) asynchronously on a given server
        
        qm = [v[0] for k,v in querymap.items()]#change the format of querymap so it matches what the subq_to_sparql function expects
        sparql_queries = server.subqueries_to_sparql(qm)
        if(len(sparql_queries)==1): # one query
            q = sparql_queries[0]
            # execute query, get response as JSON
            j = await server.sparqlQuery(q)             #asynchronous query happens here / await
            dlist = Server_proxy.makedicts(j)
        else:                   # multiple queries
            dlist=[]
            for sq in sparql_queries:
                ji = await server.sparqlQuery(sq)       #asynchronous query happens here / await
                # merge all of these json results
                dlist = dlist + Server_proxy.makedicts(ji)
        reached_nodes = []
        graphsolutions, queue, edgelist = set(), [], []
        for edge_id in querymap.keys():
            subq, reached_state = querymap[edge_id]
            #info for this rule:
            node1, is_var1, node2, is_var2 = subq.get_vars()
            
            #remove ? marks from variable names
            node1v = node1[1:] if(is_var1) else node1[1:-1]
            node2v = node2[1:] if(is_var2) else node2[1:-1]
            # get pairs for the considered edge
            pairs = [(d.get(node1v, node1v), d.get(node2v, node2v)) for d in dlist if (node1v in d or node2v in d)]
            for pp in pairs:
                reached_nodes.append((pp[1],reached_state))
                edgelist.append((pp[0], edge_id, pp[1]))
        for vg, va in reached_nodes:
            if (va.is_end):# in qa_end_states):
                graphsolutions.add(vg)
                #if (va != first_state or not(vg.startswith(domain))): #only continue exploring if we reached this node from a different server or a different state
                if (not(vg.startswith(server.domain))): #only continue exploring if we reached this node from a different server
                    queue.append((vg,va))
            else:
                queue.append((vg,va)) # enqueue for next query

        return graphsolutions, queue, edgelist


    def iterative_strategy_1sq(self, start_node, regex, logfile=None, one_query=True, prefixes=None):

        '''
        Running the product automaton algorithm on the generalized query automaton, constructed and searched on the fly.
        Asynchronous version!
        '''

        if(logfile):
            lf = open(logfile, "w")

        #reset time for logging
        tt= time.time()
        self.start_time = tt

        if (not self.knownServers):
            print("List of servers to query has not been set.\n Use method client.initiate([server1, server2...]) with Serveur objects.")
            return
        for s in self.knownServers.values():
            if(prefixes):
                s.set_query_prefixes(prefixes)
            if(not one_query):
                s.one_query=False
            if(logfile):
                s.set_logfile(lf) # sets all log files
            s.set_debugging(s.debug, tt)


        self.start_node = start_node
        self.expand_re(regex) #build generalized query automaton
        rules, qa_start_state, qa_end_states = self.expanded_re
        decomp=rules
        #print("===== decomposed REGEX ======")
        for k in decomp:
            rule = decomp[k]
        #    print(rule[0], rule[0].name, rule[1].name, rule[2][:50])
            #print(rule[0].name, rule[1].name, rule[2].sparql_str())
        #print("start state",qa_start_state.name)
        #print("end states",[q.name for q in qa_end_states])
        def get_regex(id):
            return rules[id][2]
        def get_to_state(id):
            return rules[id][1]
        NFA = self.get_NFA() #query automaton, where transitions are identifiers from above 'rules'
        #print("------ query generalized automaton")
        #NFA.uglyprint()
        #print("qa end states:", [q.name for q in qa_end_states])

        visited = set()
        queue= [(self.start_node, NFA.start)]
        edgelist = []  # list of traversed edges
        graphsolutions = set()

        loop = asyncio.get_event_loop()

        while queue:
            first_state = queue[0][1]
            next_query_vertices = set()
            next_queue = []
            for vp in queue: #vp = pair (graph node, autom state)
                if (vp in visited):
                    continue
                if (vp[1] == first_state):  #select all nodes related to the same automaton state
                    next_query_vertices.add(vp)
                    visited.add(vp)
                else:
                    next_queue.append(vp)
            queue = next_queue #enqueue the rest for next round            

            #---------- We have first_state = state of query automaton that we're exploring from
            #                   next_query_vertices = list of graph nodes that we want to connect from
            # we're exploring simultaneously a large number of vertices in product automaton, all the unexplored ones associated with a given automaton state
            
            # if vautom is end state of automaton, add graph nodes to solution list
            #if (first_state.is_end):
            #    graphsolutions.extend(next_query_vertices)
                    
            # (no epsilon-transitions in this situation: we have a deterministic but incomplete generalized automaton)
            assert(len(first_state.epsilon)==0)

            # prepare a graph query for each transition from the current automaton state
            nql = sorted(next_query_vertices,key=lambda p: self.get_URI_domain(p[0])) #sort necessary for groupby: sorting toseparate the relevant nodes for each server

            # build queries for next round of processing
            serverqueries = []
            for domain, nodes in groupby(nql, lambda p: self.get_URI_domain(p[0])):
                server = self.getServerByDomain(domain)
                qn = [n for n,s in nodes]
                querymap = {}
                #select relevant subqueries
                for ident,to_state in first_state.transitions.items():
                    regex = get_regex(ident)
                    subquery = server.prepare_subquery(first_state, to_state, regex, qn, None, ident)
                    reached_state = get_to_state(ident)
                    querymap[ident] = (subquery, to_state)
                serverqueries.append((server, querymap)) #tuple


            #run queries asynchronously

            
            tasks = []
            for server, querymap in serverqueries:
                tasks.append(loop.create_task(self.runsubqueries_async(server, querymap)))
            if(len(tasks)>0):
                finished, unfinished = loop.run_until_complete(asyncio.wait(tasks))
                list_results = map(lambda task: task.result(), finished)
                for (solutions, to_enqueue, elist) in list_results:
                    graphsolutions.update(solutions)
                    queue.extend(to_enqueue)
                    edgelist.extend(elist)

            
        if(logfile):
            lf.close()


        return graphsolutions, edgelist#)), broadcasts  # set of graph nodes in terminal nodes of product automaton; list of visited nodes; list of traversed edges; list of broadcast queries



# #----------
#     def iterative_strategy_1sq2(self, start_node, regex, logfile=None, one_query=True, prefixes=None):

#         '''
#         Running the product automaton algorithm on the generalized query automaton, constructed and searched on the fly.
#         Asynchronous version!
#         '''

#         if(logfile):
#             lf = open(logfile, "w")

#         #reset time for logging
#         tt= time.time()
#         self.start_time = tt

#         if (not self.knownServers):
#             print("List of servers to query has not been set.\n Use method client.initiate([server1, server2...]) with Serveur objects.")
#             return
#         for s in self.knownServers.values():
#             if(prefixes):
#                 s.set_query_prefixes(prefixes)
#             if(not one_query):
#                 s.one_query=False
#             if(logfile):
#                 s.set_logfile(lf) # sets all log files
#             s.set_debugging(s.debug, tt)


#         self.start_node = start_node
#         self.expand_re(regex) #build generalized query automaton
#         rules, qa_start_state, qa_end_states = self.expanded_re
#         decomp=rules
#         #print("===== decomposed REGEX ======")
#         for k in decomp:
#             rule = decomp[k]
#         #    print(rule[0], rule[0].name, rule[1].name, rule[2][:50])
#             #print(rule[0].name, rule[1].name, rule[2].sparql_str())
#         #print("start state",qa_start_state.name)
#         #print("end states",[q.name for q in qa_end_states])
#         def get_regex(id):
#             return rules[id][2]
#         def get_to_state(id):
#             return rules[id][1]
#         NFA = self.get_NFA() #query automaton, where transitions are identifiers from above 'rules'
#         #print("------ query generalized automaton")
#         #NFA.uglyprint()
#         #print("qa end states:", [q.name for q in qa_end_states])

#         visited = set()
#         queue= [(self.start_node, NFA.start)]
#         edgelist = []  # list of traversed edges
#         graphsolutions = set()

#         loop = asyncio.get_event_loop()

#         while queue:
#             first_state = queue[0][1]
#             next_query_vertices = set(filter(lambda v: v not in visited, queue))
#             visited.update(next_query_vertices)
#             queue = [] 
#             # next_queue = []
#             # for vp in queue: #vp = pair (graph node, autom state)
#             #     if (vp in visited):
#             #         continue
#             #     if (vp[1] == first_state):  #select all nodes related to the same automaton state
#             #         next_query_vertices.add(vp)
#             #         visited.add(vp)
#             #     else:
#             #         next_queue.append(vp)
#             # queue = next_queue #enqueue the rest for next round            

#             #---------- We have first_state = state of query automaton that we're exploring from
#             #                   next_query_vertices = list of graph nodes that we want to connect from
#             # we're exploring simultaneously a large number of vertices in product automaton, all the unexplored ones associated with a given automaton state
            
#             # if vautom is end state of automaton, add graph nodes to solution list
#             #if (first_state.is_end):
#             #    graphsolutions.extend(next_query_vertices)
                    
#             # (no epsilon-transitions in this situation: we have a deterministic but incomplete generalized automaton)
#             #assert(len(first_state.epsilon)==0)

#             # prepare a graph query for each transition from the current automaton state
#             nql = sorted(next_query_vertices,key=lambda p: self.get_URI_domain(p[0])) #sort necessary for groupby: sorting toseparate the relevant nodes for each server
#             #print("NQL--------", [(n,q.name) for n,q in nql])
#             # build queries for next round of processing
#             serverqueries = []
#             for domain, nodes in groupby(nql, lambda p: self.get_URI_domain(p[0])):
#                 server = self.getServerByDomain(domain)
#                 nodessort = sorted(nodes, key=lambda ns:ns[1].name) # [n for n,s in nodes]
#                 querymap = {}
#                 for first_state, snodes in groupby(nodessort, lambda ns:ns[1]):
#                     gn = [n for n,s in snodes] #graph nodes only
#                     #select relevant subqueries
#                     for ident,to_state in first_state.transitions.items():
#                         regex = get_regex(ident)
#                         #print("GN-"+first_state.name, gn)
#                         subquery = server.prepare_subquery(first_state, to_state, regex, gn, None, ident)
#                         reached_state = get_to_state(ident)
#                         querymap[ident] = (subquery, to_state)
#                 serverqueries.append((server, querymap)) #tuple


#             #run queries asynchronously

            
#             tasks = []
#             for server, querymap in serverqueries:
#                 tasks.append(loop.create_task(self.runsubqueries_async(server, querymap)))
#             if(len(tasks)>0):
#                 finished, unfinished = loop.run_until_complete(asyncio.wait(tasks))
#                 list_results = map(lambda task: task.result(), finished)
#                 for (solutions, to_enqueue, elist) in list_results:
#                     graphsolutions.update(solutions)
#                     queue.extend(to_enqueue)
#                     edgelist.extend(elist)

            
#         if(logfile):
#             lf.close()


#         return graphsolutions, edgelist#)), broadcasts  # set of graph nodes in terminal nodes of product automaton; list of visited nodes; list of traversed edges; list of broadcast queries


