from RPQ import loadgraph, runquery, compile, bfs, makeParseTree
from parse import NFA
from jinja2 import Environment, FileSystemLoader
import json
from urllib import parse, request
from functools import reduce


file_loader = FileSystemLoader("SPARQL-Templates")
env = Environment(loader=file_loader)

def handle_innodes(innodes):
    #make string in brackets, each node URI in<> + separated by commas
    return "("+ ", ".join(["<"+nd+">" for nd in innodes]) +")"

class Subquery:
    """
    Convenience class to handle SPARQL subqueries
    """
    def __init__(self, regex):
        self.regex = regex
        self.filters = []

    # def make_left_variable(yes_or_no):
    #     self.left_is_var = yes_or_no #indicates that left side is a variable, if False, it's a fixed node (single-source queries)

    # def make_right_variable(yes_or_no):
    #     self.right_is_var = yes_or_no #indicates that right side is a variable, if False, it's a fixed node (future use = boolean queries)

    def add_innode_filter(self, var, innodes):
        self.addfilter(var+" IN "+innodes)

    def add_outnode_filter(self,var, domain):
        self.addfilter("isURI("+var+") && !STRSTARTS(STR("+var+"),'"+domain+"')")

    def addfilter(self, f):
        self.filters.append(f)

    def setStartNode(self,n):
        if(n.startswith("http")):
            n = "<"+n+">"
        self.node1 = n

    def setEndNode(self,n):
        if(n.startswith("http")):
            n = "<"+n+">"
        self.node2 = n

    def to_sparql(self):
        if(self.filters):
            filterstring = " FILTER(" + " && ".join(self.filters) +")"
            query_elements = ["{", self.node1, self.regex, self.node2,".", filterstring, "}"]
        else:
            query_elements = ["{", self.node1, self.regex, self.node2, ".", "}"]
        return " ".join(query_elements)

    def as_standalone_SPARQL(self):
        return "SELECT * WHERE " + self.to_sparql()

    def get_vars(self):
        return tuple([self.node1, self.node1.startswith('?'), self.node2, self.node2.startswith('?')])

class Serveur:
    def __init__(self, name, domain, graph):
        self.name = name
        self.domain = domain
        self.graph = graph # File containing datagraph
        self.innodes = set()
        self.outnodes = set()
        self.data = {}
        self.log = False

    def set_logging(self):
        self.log=True

    def set_query_prefixes(self,pref:dict):
        if(self.log):
            print("setting prefixes to:", pref)
        self.prefixes = pref

    def set_innodes(self, nodes):
        '''
        "tell the server" which nodes are "in nodes" (nodes of local domain, duplicated on other servers)
        '''
        self.innodes = nodes


    def get_outgoing_nodes(self):
        '''
        Get all outgoing node from the server/file
        :return: a set of nodes
        '''
        if (self.outnodes): 
            return self.outnodes
        g = loadgraph(self.graph)
        domain_name = self.domain  #list(g.keys())[0].split("|")[0]  # use the first node in the file to get the domain
        nodes_out = set()
        #node_in = domain_name.lower()
        for key in g.keys():
            for value in g[key]:
                if not value[0].startswith(domain_name):
                    nodes_out.add(value[0])
        self.outnodes = nodes_out
        if (self.log):
            print("---- server ", self.name, " outgoing nodes:", nodes_out)
        return nodes_out


    def get_local_response(self, query_automaton, start_node):
        '''
        Simulates the server, getting local responses to a query
        Returns a tuple with a data structure with responses to all the requests sent to the servers
        and a set of not filtered nodes
        :param expanded_re: a dict of rules {rule : start_state, end_state, regex}
        :return: Tuple of a dict of {rule: [(origin_node, {response nodes})]} and a set of not filtered nodes
        '''
        # TODO Use in_nodes and expanded_re to get responses from the server
        expanded_re, qa_start_state, qa_end_states = query_automaton
        data_graph = dict()
        
        for rule in expanded_re:
            #info for this rule:
            from_state, to_state, regex = expanded_re[rule]

            data_graph.setdefault(rule, [])
            if (from_state == qa_start_state): 
                if (start_node.startswith(self.domain)):
                    from_nodes = [start_node]#single-source query
                    if (self.log):
                        print("-- single-source query from ", start_node, "with regex:", regex)
                else:
                    continue #nothing to do for this rule, start node not in this graph
            else: # multi-source query from all in_nodes
                from_nodes = self.innodes
                if (self.log):
                    print("-- multi-source query from ", from_nodes, "with regex:", regex)
            
            for node in from_nodes:
                #run multi-source query = single-source for each from node
                sol, visited, edgelist, bc = runquery(loadgraph(self.graph), node, regex)
                if (self.log):
                    print("-> potential answers", sol)

                res_nodes = set(filter(lambda n: (n in self.outnodes or to_state.is_end),sol))
                # for res_node in sol:
                #     if (res_node in self.outnodes or to_state.is_end): 
                #         res_nodes.add(res_node)
                #         if expanded_re[rule][1].is_end:
                #             not_filtered.add(res_node)
                data_graph[rule] = data_graph[rule] + [(node, rn) for rn in res_nodes]
                # if len(res_nodes) != 0:
                #     data_graph[rule].append((node, res_nodes))

        if (self.log):
            print ("server", self.name, "response:--------------------\n", data_graph)
        return data_graph

class Server_proxy(Serveur):
    def __init__(self, name, uri, domain):
        self.name = name
        self.domain = domain
        self.uri = uri # full URI to connect to this server (ip:port?)
        self.innodes = set()
        #self.outnodes = set() # do I need this?
        self.data = {}
        self.one_query = True
        self.log=False

    def get_local_response(self, query_automaton, start_node): #overrides this for a proxy type server
        '''
        Gets responses from each server to build data graph
        Returns a tuple with a data structure with responses to all the requests sent to the servers
        :param expanded_re: a dict of rules {rule : start_state, end_state, regex}
        :return: Tuple of a dict of {rule: [(origin_node, {response nodes})]} and a set of not filtered nodes
        '''
        # TODO Use in_nodes and expanded_re to get responses from the server
        subqueries = self.prepare_query2(query_automaton,start_node)

        if(self.one_query):
            q = "SELECT * WHERE \n{" + "\nUNION\n".join([s.to_sparql() for s in subqueries.values()]) +"}"
            if(self.prefixes):
                q = "PREFIX "+ "\n".join([k+":"+self.prefixes[k] for k in self.prefixes]) + "\n" +q
            # execute query, get response as JSON
            j = self.sparqlQuery(q)
            dlist = Server_proxy.makedicts(j)
        else:
            dlist=[]
            for k,sq in subqueries.items():
                q = "SELECT * WHERE \n"+sq.to_sparql()
                if(self.prefixes):
                    q = "PREFIX "+ "\n".join([k+":"+self.prefixes[k] for k in self.prefixes]) + "\n" +q
                ji = self.sparqlQuery(q)
                # merge all of these json results
                dlist = dlist + Server_proxy.makedicts(ji)
                

                
        

        data_graph = dict()
        not_filtered = set()
        for edge_id,subq in subqueries.items():
            #info for this rule:
            node1, is_var1, node2, is_var2 = subq.get_vars()
            
            #remove ? marks from variable names
            node1v = node1[1:] if(is_var1) else node1[1:-1]
            node2v = node2[1:] if(is_var2) else node2[2:-2]
            # get pairs for the considered edge
            pairs = [(d.get(node1v, node1v), d.get(node2v, node2v)) for d in dlist if (node1v in d or node2v in d)]
            data_graph[edge_id] = pairs
                        
        if (self.log):
            print ("server", self.name, "response:--------------------\n", data_graph)
        return data_graph

    def prepare_query2(self, query_automaton, start_node):
        '''
        This function scans through an expanded query automaton. For every rule, it examines what its starting and final nodes are,
        iteratively creating a complete and valid SPARQL query.

        '''
        domain = self.domain  # server domain name
        innodes = handle_innodes(self.innodes) # server incoming nodes
        
        rules, qa_start_state, qa_end_states = query_automaton

        entrantx = 1  # counter for number of incoming nodes
        rx = 1  # counter for rules with starting node
        sortantx = 1
        ruleCounter = 0  # counter for number of rules for number of unions to add
        finalRule = len(rules)  # total number of rules given
        subqueries = {}
        for r in rules:
            #info for this rule:
            from_state, to_state, regex = rules[r]
            t = makeParseTree(regex)
            if (self.log):
                print('============== regex parsing: for states',from_state.name, to_state.name, regex)
                print(str(t))
                t.printIndented(2)
                print(t.sparql_str())

            edge_id = r #rule identifier from_state.name+'_'+to_state.name
            subquery = Subquery(t.sparql_str()) #create new subquery
            left_is_incoming = not (from_state.name == qa_start_state.name) 
            right_is_outgoing = not (to_state.is_end) #TODO: additional analysis indicating if nodes may be in/out as well (inverse operators)
            if(self.log):
                print("left is an incoming node:",left_is_incoming, from_state.name , qa_start_state.name)
                print("right is an outgoing node:",right_is_outgoing,to_state.name)

            #start node:
            if(not left_is_incoming):
                subquery.setStartNode(start_node)
            else:
                var_left= "?"+edge_id+"_f"
                subquery.setStartNode(var_left)
                subquery.add_innode_filter(var_left, innodes)

            #end node
            var_right = "?"+edge_id+"_r"
            subquery.setEndNode(var_right)
            if (right_is_outgoing): # If leads into an outnode
                subquery.add_outnode_filter(var_right, domain)
            #map edge id to subquery object
            subqueries[edge_id]=subquery

        # query = "SELECT * WHERE { \n"
        # for sq in subqueries:

        # query += "\n\n}"
        return subqueries

    def make_sparql_query_outgoing_nodes(self):
        '''
        prepare a SPARQL query to retrieve all outgoing nodes for this server
        :return: the SPARQL query
        '''
        # Print the get outnodes SPARQL request
        template = env.get_template("outnodes-template.j2")
        temp_render = template.render(domain=self.domain)

        return temp_render

    def get_outgoing_nodes(self):
        '''
        Get all outgoing node from the server/file
        :return: a set of nodes
        '''
        q = self.make_sparql_query_outgoing_nodes()
        j = self.sparqlQuery(q)
        dlist = Server_proxy.makedicts(j)
        return [d['c'] for d in dlist] #the query template uses ?c as variable for outgoing nodes
 

    def makedicts(jsonresponse):
        """
        extracts variable bindings from json response
        returns list of dictionaries {var: value}
        """
        vars = jsonresponse['head']['vars']
        bindings = jsonresponse['results']['bindings']
        dictlist = list(map(lambda bind: {v:bind[v]['value'] for v in vars if v in bind}, bindings))
        return dictlist

    def sparqlQuery(self, query, format="application/json"):
        """
        sends a SPARQL query to the proxied server and returns results as JSON
        """
        #if(self.log):
        print("\n---- sending SPARQL query to server: ", self.name, "\n")
        print(query)

        baseURL = self.uri
        
        params={
            "default-graph": "",
            "should-sponge": "soft",
            "query": query,
            "debug": "on",
            "timeout": "",
            "format": format,
            "save": "display",
            "fname": ""
        }
        querypart= parse.urlencode(params).encode("utf-8")  
        req = request.Request(baseURL)
        with request.urlopen(req,data=querypart) as f:
            resp = f.read()
            data= json.loads(resp)
            print ("\n---- Server Response: --- \n",data, "\n")
            return data


        

class Client:
    def __init__(self, name):
        self.name = name
        self.knownServers = None

    def expand_re(self, regex):
        '''
        Uses a regular expression in a format like <a><b>*<c> and decomposes it
        :param regex: regular expression to decompose
        :return: nothing; the expanded query automaton is added as an attribute to the Client object.
        '''

        NFA1 = compile(regex)
        DFA = NFA1.toDFA()
        DFA.renameStates()
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


    def get_data_responses(self, server):
        '''
        Sends the query to the server object using the Serveur.get_server_response method
        :param server: Serveur instance used to send the query
        :return: a dict format -> {rule : [(start_node, {set of end nodes})] and a set of not filtered nodes
        '''
        #er_list, start_state, end_states = self.expanded_re
        response = server.get_local_response(self.expanded_re, self.start_node) #gets response using "simulated" server functionality

        return response

    def get_data_graph(self):
        '''
        Get all the responses data structure from all the servers and merge then into a single graph
        :param graph_list: a list of all the graph where requests are sent
        :param origin_er: regular expression to use to find the endpoints
        :return: a dict of all the data in a graph format
        '''

        serv_list = list(self.knownServers.values())
        list_results = map(lambda s: self.get_data_responses(s), serv_list)

        def merge_reducer(d1, d2):
            keys = set(d1.keys()).union(d2.keys())
            return {k: d1.get(k, []) + d2.get(k, []) for k in keys} #merged dictionaries

        full_result = reduce(merge_reducer, list_results)

        # Format the data_responses merged into a graph using the loadgraph format
        new_graph = dict()

        for key in full_result:
            for nodepair in full_result[key]:
                node1, node2 = nodepair
                label = key
                new_graph.setdefault(node1, []).append((node2, label))
                new_graph.setdefault(node2, [])

        return new_graph

    # def get_server_out_nodes(self, server):
    #     '''
    #     Returns the outgoing nodes of a specified server, if said server exists in the knownServers list
    #     Otherwise, adds this server to the list of knownServers
    #     :param server: a single server for which you want the outgoing nodes

    #     #TODO: figure out the use of this
    #     '''
    #     outnodes = server.get_outgoing_nodes()
    #     if server.name not in self.knownServers:
    #         self.knownServers.update({server.name : (server.domain, {"outnodes": (outnodes)}, {"innodes" : ()})})
    #     else:
    #         self.knownServers[server.name][1]["outnodes"] = outnodes

    def get_innodes(self, server, outnodes_list):
        '''
        Gets all incoming nodes for a given server.
        :param server: The server for which you want to find the incoming nodes
        :param outnodes_list: Outnodes list for outside servers of the one you inquire about
        :return: Updates this client's knownServers innodes for the server inquired about
        '''
        server_nodes = set(filter(lambda n: n.startswith(server.domain), outnodes_list))
        server.set_innodes(server_nodes) # Update known incoming nodes list for specified server

    # def get_all_out_nodes(self, list_of_servers):
    #     '''
    #     Use a list of filenames (graphs) to get all the outgoing nodes
    #     Returns a set of all outgoing nodes
    #     '''
    #     all_out_nodes = set()
    #     for server in list_of_servers:
    #         outnodes = server.get_outgoing_nodes_file()
    #         all_out_nodes.update(outnodes)
    #     return all_out_nodes

    def get_NFA(self):
        '''
        Builds an NFA using the State and NFA classes from parse.py
        :return: an NFA object
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
            transitions.setdefault(rules[rule][0], set())
            transitions[rules[rule][0]].add((rules[rule][1], rule))

        # Transform into a set of transition in dict and add them to the actual state
        for state in transitions:
            actual_transitions = {}
            for transition in transitions[state]:
                actual_transitions.setdefault(transition[1], transition[0])
            state.transitions = actual_transitions

        # Get first state


        start_state = self.expanded_re[1]

        list_of_states.remove(start_state)

        # Get last states
        for state in list_of_states:
            if state.is_end:
                end_state = state
                list_of_states.remove(state)

        # Set start and end states

        result_NFA = NFA(start_state)

        # Add all the states in between
        for inter_state in list_of_states:
            result_NFA.addstate(inter_state, set())

        return result_NFA

    def set_servers_in_out_nodes(self):
        '''
        Sets the all the outnodes and innodes for all the Knownservers of the Client instance with the given responses
        :param list_of_servers: List of Serveur instances for the given Client
        :return: None
        '''
        # Add all outnodes for each client.knownservers
        all_out_nodes = set()
        for server in self.knownServers.values():
            outnodes = server.get_outgoing_nodes()
            all_out_nodes.update(outnodes)
        
        all_innodes = {name: [] for name in self.knownServers}

        #assign each node to its server/domain
        for node in all_out_nodes:
            for name in all_innodes:
                if (node.startswith(self.knownServers[name].domain)):
                    all_innodes[name].append(node)

        if(self.log):
            print("------ incoming nodes --", all_innodes)
        # tell each server which are its innodes
        for name in self.knownServers:
            #TODO: temp solution: add start_node as incoming node to its server
            #if(self.start_node.startswith(self.knownServers[name].domain)):
            #    all_innodes[name].append(self.start_node)
            
            self.knownServers[name].set_innodes(all_innodes[name])



    def initiate(self, list_of_servers, logging=False):
        '''
        Sets all the used data in the Client for the next steps
        :param list_of_servers: sets the list of servers in the Client object
        :param regex: string regex used to set the expanded regex in the Client object
        :param start_node: string node added to the innodes of the starting server
        :return:
        '''
        self.knownServers = {s.name: s for s in list_of_servers} #map each server name to the server object
        self.log = logging
        if(logging):
            for s in list_of_servers:
                s.set_logging()
        

    def run_single_source_query(self, start_node, regex, prefixes=None, one_query=True):
        '''
        Main method of the program
        Run a distributed query on the list of servers then run a local query on the resulting local graph
        :return:
        '''
        if (not self.knownServers):
            print("List of servers to query has not been set.\n Use method client.initiate([server1, server2...]) with Serveur objects.")
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
        self.set_servers_in_out_nodes()

        # Get local graph with all the responses
        responses = self.get_data_graph()
        if(self.log):
            print("------ data graph: [generalized automaton]", responses)
        # Create NFA with the expanded regex
        temp_NFA = self.get_NFA()

        # Run query on the local graph with the expanded regex NFA
        sol, v, e, b = bfs(responses, temp_NFA, self.start_node) #function returns a 4-tuple, here we only care about 1st element
        return sol


