from RPQ import runquery, compile, bfs, makeParseTree
from parse import NFA
from jinja2 import Environment, FileSystemLoader
import json, re, time
from urllib import parse, request
from functools import reduce
from itertools import groupby
from enum import Enum
import asyncio
import aiohttp

file_loader = FileSystemLoader("SPARQL-Templates")
env = Environment(loader=file_loader)

class QueryConstraint(Enum):
    ANY = 1
    NODE_OR_NOTHING = 2
    NODE_OR_BORDER = 3
    BORDER = 4

class Subquery:
    """
    Convenience class to handle SPARQL subqueries
    """
    def __init__(self, regex):
        self.regex = regex
        self.filters = {} #keyed by variable that they apply to... then combine each with OR , then combine with AND across variables
        self.innodes = None
        self.outfilter_domain= None

    # def make_left_variable(yes_or_no):
    #     self.left_is_var = yes_or_no #indicates that left side is a variable, if False, it's a fixed node (single-source queries)

    # def make_right_variable(yes_or_no):
    #     self.right_is_var = yes_or_no #indicates that right side is a variable, if False, it's a fixed node (future use = boolean queries)

    def add_innode_filter(self, var, innodes):
        self.innodes = innodes
        self.addfilter(var, var+" IN ("+", ".join(["<"+nd+">" for nd in innodes]) +")")

    def add_outnode_filter(self,var, domain):
        self.addfilter(var, "( isURI("+var+") && !STRSTARTS(STR("+var+"),'"+domain+"'))")
        self.outfilter_domain = domain

    def add_node_filter(self, var, node):
        self.addfilter(var, var+"=<"+node +">")

    def addfilter(self, var, f):
        self.filters.setdefault(var,[]).append(f)

    def setStartNode(self,n):
        if(n.startswith("http")):
            self.innodes = [n]
            n = "<"+n+">"
        self.node1 = n

    def setEndNode(self,n):
        if(n.startswith("http")):
            n = "<"+n+">"
        self.node2 = n

    def to_sparql(self):
        if(self.filters):
            allvarfilters = ["||".join(flist) for v,flist in self.filters.items()]
            if (len(allvarfilters)>1):
                joined = "(" + ") && (".join(allvarfilters) +")"
            else:
                joined = allvarfilters[0]
            filterstring = " FILTER(" + joined +")"
        else:
            filterstring = ""
        query_elements = ["{", self.node1, self.regex.sparql_str(), self.node2,".", filterstring, "}"]
        
        return " ".join(query_elements)

    def as_standalone_SPARQL(self):
        return "SELECT * WHERE " + self.to_sparql()

    def get_vars(self):
        return tuple([self.node1, self.node1.startswith('?'), self.node2, self.node2.startswith('?')])

class Serveur:
    def __init__(self, name, domain, graph_file):
        self.name = name
        self.domain = domain
        self.loadgraph(graph_file)
        self.innodes = set()
        self.outnodes = set()
        self.data = {}
        self.debug = False
        #self.start_time # set by client

    def set_debugging(self, debugging, t):
        self.debug=debugging
        self.start_time = t

    def set_query_prefixes(self,pref:dict):
        if(self.debug):
            print("setting prefixes to:", pref)
        self.prefixes = pref

    def set_innodes(self, nodes):
        '''
        "tell the server" which nodes are "in nodes" (nodes of local domain, duplicated on other servers)
        '''
        self.innodes = nodes
        if(self.debug):
            print(self.name, "innodes set to:", self.innodes)

    def loadgraph(self,filename):
        if filename.endswith(".nt"):
            self.graph = self.loadgraphNT(filename)
        elif filename.endswith(".txt"):
            self.graph = self.loadgraphTxt(filename)

    def loadgraphTxt(self, gfname):
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
            #if (cnt % 10000 == 0): print(cnt)
            if (len(line) <= 1): continue
            tup = line.split()
            node1, node2, label = tup[0], tup[1], tup[2]
            thegraph.setdefault(node1, []).append((node2, label))
            thegraph.setdefault(node2, [])
        grafile.close()
        return thegraph


    def loadgraphNT(self, gfname):
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


    def get_outgoing_nodes(self, regex):
        '''
        Get all outgoing node from the server/file
        :return: a set of nodes
        '''
        # do it on local graph
        labels =  [s[1:-1] for s in set(re.findall(r'<.+?>', regex))]
        allnodes = set()
        for node1 in self.graph:
            allnodes.update([n2 for (n2,ll) in self.graph[node1] if (ll in labels and not n2.startswith(self.domain))])
        if(self.debug):
            print("outgoing nodes: [server", self.name,"]", len(allnodes))
        self.outnodes=allnodes
        return allnodes


    def get_local_response(self, query_automaton, start_node=None, end_node=None):
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

            
            if(start_node):
                if (from_state == qa_start_state): 
                    if (start_node.startswith(self.domain)):
                        from_nodes = [start_node]#single-source query
                        if (self.debug):
                            print("-- single-source query from ", start_node, "with regex:", regex)
                    else:
                        continue #nothing to do for this rule, start node not in this graph
                else: # multi-source query from all in_nodes
                    from_nodes = self.innodes
                    if (self.debug):
                        print("-- multi-source query from ", from_nodes, "with regex:", regex)
            else: # no start node given... multi-source query from all nodes
                from_nodes = list(self.graph.keys())

            for node in from_nodes:
                #run multi-source query = single-source for each from node
                sol, visited, edgelist, bc = runquery(self.graph, node, regex)
                if (self.debug):
                    print("-> potential answers", sol)
                #filter condition on potential solutions: not end state => outgoing nodes only, end state=> any node works (non end nodes may be relevant as they may lead to the end node in other end states)
                res_nodes = filter(lambda n: ((n in self.outnodes and not to_state.is_end) or to_state.is_end),sol)

                data_graph[rule] = data_graph.setdefault(rule, set()).union(set([(node, rn) for rn in res_nodes]))

        if (self.debug):
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
        self.debug=False
        self.prefixes =[]
        self.logfile=None

    #set a new logfile. 
    def set_logfile(self,logfile):
        self.logfile = logfile

    def get_local_response(self, query_automaton, start_node=None, end_node=None): #overrides this for a proxy type server
        '''
        Gets responses from each server to build data graph
        Returns a tuple with a data structure with responses to all the requests sent to the servers
        :param query_automaton: NFA representing the query regex
        :param start_node: start_node for single-source query
        :param end_node: end_node for single-source or boolean query #TODO: not implemented
        :param dry_run: true if we just want to build the SPARQL query and print it out
        :return: Tuple of a dict of {rule: [(origin_node, {response nodes})]} and a set of not filtered nodes
        '''
        # prepare sparql + query objects (which we need in later step)
        subqueries = self.prepare_query(query_automaton,start_node, end_node)
        sparql_queries = self.subqueries_to_sparql([s for e,s in subqueries])

        if(len(sparql_queries)==1): # one query
            q = sparql_queries[0]
            # execute query, get response as JSON
            j = self.sparqlQuery(q)
            dlist = Server_proxy.makedicts(j)
        else:                   # multiple queries
            dlist=[]
            for sq in sparql_queries:
                ji = self.sparqlQuery(sq)
                # merge all of these json results
                dlist = dlist + Server_proxy.makedicts(ji)            
        

        data_graph = dict()
        for edge_id,subq in subqueries:
            #info for this rule:
            tt = subq.get_vars()
            #print("TTTTTTTTT ---------------", tt)
            node1, is_var1, node2, is_var2 = tt
            
            #remove ? marks from variable names, < > from non-variables
            node1v = node1[1:] if(is_var1) else node1[1:-1]
            node2v = node2[1:] if(is_var2) else node2[1:-1]
            # get pairs for the considered edge
            #pairs = [(d.get(node1v, node1v), d.get(node2v, node2v)) for d in dlist if (node1v in d or node2v in d)]
            pairs = set([(d.get(node1v, node1v), d.get(node2v, node2v)) for d in dlist if (node1v in d or node2v in d)])
            data_graph[edge_id] = pairs
                        
        if (self.debug):
            print ("server", self.name, "response:--------------------\n", data_graph)
        return data_graph


    def prepare_subquery(self, from_state, to_state, regex_tree, start_node, end_node, edge_id):
        t = makeParseTree(regex)
        #handle case where subquery regex is a* => replace with a+
        t.star_to_plus() #convert star to plus, or if node is alt, check children and do same.
        if (self.debug):
            print('============== subquery for states',from_state.name, to_state.name, regex, "with start nodes", start_nodes)
            #print(str(t))
            #t.printIndented(2)
            #print(t.sparql_str())

        
        var_left= "?"+edge_id+"_f"
        var_right= "?"+edge_id+"_r"
        ##########################################################
        # restrictions on nodes to retrieve on left and right sides:
        # -- Left
        # no_start_node and from_state is start => left = any
        # start_node and from_state is start and from_state.indegree==0: left = node (or nothing if node known not to be found on server [forwards edges only])
        # start_node and from_state is start and from_state.indegree>0: left = node or border
        # from_state not start [=> has indegree>0]: left= border
        # Borders:
        # if paths start with forwards, left border = incoming nodes
        # if paths start with inverse, left border = outgoing nodes
        
        # -- Right
        # no_end_node and to_state is accepting => right = any
        # end_node and to_state is accepting and to_state.outdegree==0: right = node (or nothing if node known not to be found on server [inverse edges only])
        # end_node and to_state is accepting and to_state.outdegree>0: right = node or border
        # to_state not accepting [=> has outdegree>0]: right= border
        # Borders:
        # if paths end with forwards, right border = outgoing nodes
        # if paths end with inverse, right border = incoming nodes
        #This information in code:
        if (from_state.name == qa_start_state.name): # from state is start
            if(start_node):
                if(start_state_has_predecessors):
                    left_constraint = QueryConstraint.NODE_OR_BORDER
                    left_side = var_left
                else:
                    left_constraint = QueryConstraint.NODE_OR_NOTHING
                    if (start_node.startswith(domain) or regex_tree.canBeginInverse()): #TODO: can begin inverse and startnode is replicated
                        left_side = start_node
                    else: #if start node is not on this server, we can cancel this subquery
                        return None #skip to next rule/subquery
            else:
                left_constraint = QueryConstraint.ANY
                left_side = var_left
        else:
            left_constraint = QueryConstraint.BORDER
            left_side = var_left
        if (to_state.is_end): # end state is accepting
            if(end_node):
                if len(to_state.transitions)>0: #to_state outdegree>0
                    right_constraint = QueryConstraint.NODE_OR_BORDER
                    right_side = var_right
                else:
                    right_constraint = QueryConstraint.NODE_OR_NOTHING
                    if (regex_tree.canEndForwards() or end_node.startswith(domain)):
                        right_side = end_node
                    else: #if start node is not on this server, we can cancel this subquery
                        return None#continue #skip to next rule/subquery
            else:
                right_constraint = QueryConstraint.ANY
                right_side = var_right
        else:
            right_constraint = QueryConstraint.BORDER
            right_side = var_right
        #Build Subquery
        subquery = Subquery(regex_tree)
        subquery.setStartNode(left_side)
        subquery.setEndNode(right_side)
        #add filters-------------
        def makefilters(side_constraint, from_side, end_node, variable): #defining function because it's the same process on both sides
            if (side_constraint is QueryConstraint.BORDER or side_constraint is QueryConstraint.NODE_OR_BORDER):
                if(nfa_temp.has_border_condition(from_state if from_side else to_state, from_side, True)): # checking if we need outnodes on left (from state)
                    subquery.add_outnode_filter(variable, domain) 
                if(nfa_temp.has_border_condition(from_state if from_side else to_state, from_side, False)):
                    if (side_constraint is QueryConstraint.NODE_OR_BORDER): #innodes or actual endpoint node
                        nodes_filter = self.innodes+([end_node] if end_node.startswith(domain) else[])
                        subquery.add_innode_filter(variable, nodes_filter)
                    else: # just the innodes
                        subquery.add_innode_filter(variable, self.innodes)
        #now define both sets of filters with above function
        makefilters(left_constraint, True, start_node, var_left)
        makefilters(right_constraint, False, end_node, var_right)

        return subquery


    def prepare_query(self, query_automaton, start_node=None, end_node=None, log=True):
        '''
        This function scans through an expanded query automaton. For every rule, it examines what its starting and final nodes are,
        iteratively creating a complete and valid SPARQL query.
        Works for multi-source queries (start_node = None)
        :returns one or several SPARQL queries

        '''
        domain = self.domain  # server domain name
        
        rules, qa_start_state, qa_end_states = query_automaton

        #figure out if start state has predecessors:
        nfa_temp = NFA(qa_start_state)
        start_state_has_predecessors = (nfa_temp.indegree(qa_start_state)>0)

        #finalRule = len(rules)  # total number of rules given
        subqueries = []
        for edge_id in rules:
            #info for this rule:
            from_state, to_state, regex = rules[edge_id]
            regex_tree = makeParseTree(regex).non_eps()
            #subquery = self.prepare_subquery(from_state, to_state, regex_tree, start_node, end_node, edge_id)
            var_left= "?"+edge_id+"_f"
            var_right= "?"+edge_id+"_r"
            ##########################################################
            # restrictions on nodes to retrieve on left and right sides:

            # -- Left
            # no_start_node and from_state is start => left = any
            # start_node and from_state is start and from_state.indegree==0: left = node (or nothing if node known not to be found on server [forwards edges only])
            # start_node and from_state is start and from_state.indegree>0: left = node or border
            # from_state not start [=> has indegree>0]: left= border
            # Borders:
            # if paths start with forwards, left border = incoming nodes
            # if paths start with inverse, left border = outgoing nodes
            
            # -- Right
            # no_end_node and to_state is accepting => right = any
            # end_node and to_state is accepting and to_state.outdegree==0: right = node (or nothing if node known not to be found on server [inverse edges only])
            # end_node and to_state is accepting and to_state.outdegree>0: right = node or border
            # to_state not accepting [=> has outdegree>0]: right= border
            # Borders:
            # if paths end with forwards, right border = outgoing nodes
            # if paths end with inverse, right border = incoming nodes

            #This information in code:
            if (from_state.name == qa_start_state.name): # from state is start
                if(start_node):
                    if(start_state_has_predecessors):
                        left_constraint = QueryConstraint.NODE_OR_BORDER
                        left_side = var_left
                    else:
                        left_constraint = QueryConstraint.NODE_OR_NOTHING
                        if (start_node.startswith(domain) or regex_tree.canBeginInverse()): #TODO: can begin inverse and startnode is replicated
                            left_side = start_node
                        else: #if start node is not on this server, we can cancel this subquery
                            continue #skip to next rule/subquery

                else:
                    left_constraint = QueryConstraint.ANY
                    left_side = var_left
            else:
                left_constraint = QueryConstraint.BORDER
                left_side = var_left


            if (to_state.is_end): # end state is accepting
                if(end_node):
                    if len(to_state.transitions)>0: #to_state outdegree>0
                        right_constraint = QueryConstraint.NODE_OR_BORDER
                        right_side = var_right
                    else:
                        right_constraint = QueryConstraint.NODE_OR_NOTHING
                        if (regex_tree.canEndForwards() or end_node.startswith(domain)):
                            right_side = end_node
                        else: #if start node is not on this server, we can cancel this subquery
                            continue #skip to next rule/subquery
                else:
                    right_constraint = QueryConstraint.ANY
                    right_side = var_right
            else:
                right_constraint = QueryConstraint.BORDER
                right_side = var_right

            #Build Subquery
            subquery = Subquery(regex_tree)
            subquery.setStartNode(left_side)
            subquery.setEndNode(right_side)
            #add filters-------------
            def makefilters(side_constraint, from_side, end_node, variable): #defining function because it's the same process on both sides
                if (side_constraint is QueryConstraint.BORDER or side_constraint is QueryConstraint.NODE_OR_BORDER):
                    if(nfa_temp.has_border_condition(from_state if from_side else to_state, from_side, True)): # checking if we need outnodes on left (from state)
                        subquery.add_outnode_filter(variable, domain) 
                    if(nfa_temp.has_border_condition(from_state if from_side else to_state, from_side, False)):
                        if (side_constraint is QueryConstraint.NODE_OR_BORDER): #innodes or actual endpoint node
                            nodes_filter = self.innodes+([end_node] if end_node.startswith(domain) else[])
                            subquery.add_innode_filter(variable, nodes_filter)
                        else: # just the innodes
                            subquery.add_innode_filter(variable, self.innodes)
            #now define both sets of filters with above function
            makefilters(left_constraint, True, start_node, var_left)
            makefilters(right_constraint, False, end_node, var_right)

            #done filters-----------
            subqueries.append((edge_id,subquery))

        #at this point we have a list of pairs (edge_id,subquery objects)
        
        return subqueries 


    def subqueries_to_sparql(self, subqueries):
        '''
        takes a list of subquery objects and returns a list of sparql queries (String), where the list contains a single union query if self.one_query==True
        '''
        sparql_queries= []
        if(self.one_query):
            q = "SELECT * WHERE \n{" + "\nUNION\n".join([s.to_sparql() for s in subqueries]) +"}"
            q = self.add_prefixes(q) # add the prefixes if applicable, shorten query string
            sparql_queries= [q]
        else:
            for sq in subqueries:
                q = "SELECT * WHERE \n"+sq.to_sparql()
                q = self.add_prefixes(q)
                sparql_queries.append(q)
        return sparql_queries
    
    def add_prefixes(self,q):
        '''
        adds the prefixes to the sparql query + replaces in query string
        '''
        if(self.prefixes):

            for pr in self.prefixes:
                pattern1 =  re.compile(self.prefixes[pr][:-1]+'(.*?)>')
                # this function is to properly replace the URI by the appropriate prefix + the local name with escaped charas (for now only '+')
                def replacefun(matchobj):
                    localname=matchobj.group(1)
                    return  pr+':'+localname.replace('+', '\\+')
                q = re.sub(pattern1,replacefun, q) # substitute URI prefix for abbreviation, removing <> in process
            q = "\n".join(["PREFIX "+ p+":"+self.prefixes[p] for p in self.prefixes]) + "\n" +q
        return q



    def make_sparql_query_outgoing_nodes(self, regex=None):
        '''
        prepare a SPARQL query to retrieve all outgoing nodes for this server
        #if regex is given, make query specific to edges in expression
        :return: the SPARQL query
        '''
        def forwards(uri):
            if (uri.startswith('<^')):
                return '<'+uri[2:]
            return uri
        if (regex): #regex is given!
            #get all labels
            labels =  set(map(forwards, re.findall(r'<.+?>', regex)))
            query= 'SELECT ?c \n WHERE { \n ?a ?b ?c. FILTER(isURI(?c) && !STRSTARTS(STR(?c), "'+self.domain+'") && ?b in ('
            query += ', '.join(sorted(labels))
            query += '))\n}'

        else:
            # Print the get outnodes SPARQL request
            template = env.get_template("outnodes-template.j2")
            query = template.render(domain=self.domain)

        if(self.prefixes):
            #query = "\n".join(["PREFIX "+k+":"+self.prefixes[k] for k in self.prefixes]) + "\n" +query
            query = self.add_prefixes(query)
        return query

    def get_outgoing_nodes(self, regex):
        '''
        Get all outgoing node from the server/file
        :return: a set of nodes
        '''
        q = self.make_sparql_query_outgoing_nodes(regex)
        j = self.sparqlQuery(q)
        dlist = Server_proxy.makedicts(j)
        self.outnodes = set([d['c'] for d in dlist]) #the query template uses ?c as variable for outgoing nodes
        return self.outnodes
 

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
        if(self.logfile):
            #print("logging!")
            logtxt= "\t".join(["Query-to-server-name-size-t:", self.name, str(len(query.encode('utf-8'))), str(time.time()-self.start_time)])
            self.logfile.write(logtxt+"\n")
        if(self.debug):
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
            size = len(resp)
            data= json.loads(resp)
            if(self.logfile):
                logtxt= "\t".join(["Server-Response:---name-sizebytes-t:", self.name, str(size), str(time.time()-self.start_time)])
                self.logfile.write(logtxt+"\n")
            if(self.debug):
                print(data)
            return data



class Server_proxy_async(Server_proxy):
    def __init__(self, name, uri, domain):
        self.name = name
        self.domain = domain
        self.uri = uri # full URI to connect to this server (ip:port?)
        self.innodes = set()
        #self.outnodes = set() # do I need this?
        self.data = {}
        self.one_query = True
        self.debug=False
        self.prefixes =[]
        self.logfile=None
        self.session = aiohttp.ClientSession()

    async def close_session(self):
    	await self.session.close()

    async def get_local_response(self, query_automaton, start_node=None, end_node=None): #overrides this for an async proxy type server
        '''
        Gets responses from this server to build data graph
        Returns a tuple with a data structure with responses to all the requests sent to the servers
        :param query_automaton: NFA representing the query regex
        :param start_node: start node for single-source or boolean query
        :param end_node: end node for single-source or boolean query
        :return: Tuple of a dict of {rule: [(origin_node, {response nodes})]} and a set of not filtered nodes
        '''
        # prepare sparql + query objects (which we need in later step)
        subqueries = self.prepare_query(query_automaton,start_node, end_node)
        sparql_queries = self.subqueries_to_sparql([s for e,s in subqueries]) #send only the subquery objects

        if(len(sparql_queries)==1): # one query
            q = sparql_queries[0]
            # execute query, get response as JSON
            j = await self.sparqlQuery(q)
            dlist = Server_proxy.makedicts(j)
        else:                   # multiple queries
            dlist=[]
            for sq in sparql_queries:
                ji = await self.sparqlQuery(sq)
                # merge all of these json results
                dlist = dlist + Server_proxy.makedicts(ji)            
        

        data_graph = dict()
        for edge_id,subq in subqueries:
            #info for this rule:
            node1, is_var1, node2, is_var2 = subq.get_vars()
            
            #remove ? marks from variable names
            node1v = node1[1:] if(is_var1) else node1[1:-1]
            node2v = node2[1:] if(is_var2) else node2[2:-2]
            # get pairs for the considered edge
            #pairs = [(d.get(node1v, node1v), d.get(node2v, node2v)) for d in dlist if (node1v in d or node2v in d)]
            pairs = set([(d.get(node1v, node1v), d.get(node2v, node2v)) for d in dlist if (node1v in d or node2v in d)])
            data_graph[edge_id] = pairs
                        
        if (self.debug):
            print ("server", self.name, "response:--------------------\n", data_graph)
        return data_graph


    async def get_outgoing_nodes(self, regex):
        '''
        Get all outgoing node from the server/file -- async version
        :return: a set of nodes
        '''
        q = self.make_sparql_query_outgoing_nodes(regex)
        j = await self.sparqlQuery(q)
        dlist = Server_proxy.makedicts(j)
        self.outnodes = set([d['c'] for d in dlist]) #the query template uses ?c as variable for outgoing nodes
        return self.outnodes
 

    async def sparqlQuery(self, query):
        """
        sends a SPARQL query to the proxied server and returns results as JSON - asynchronous version (coroutine)
        """
        if(self.logfile):
            #print("logging!")
            logtxt= "\t".join(["Query-to-server-name-size-t:", self.name, str(len(query.encode('utf-8'))), str(time.time()-self.start_time)])
            self.logfile.write(logtxt+"\n")
        if(self.debug):
            print(query)

        baseURL = self.uri
        
        myparams={
            "default-graph": "",
            "should-sponge": "soft",
            "query": query,
            "debug": "on",
            "timeout": "",
            "format": "application/json",
            "save": "display",
            "fname": ""
        }
        async with self.session.get(baseURL, params=myparams) as response:
            resp = await response.text()
            size = len(resp)
            data= json.loads(resp)
            
            if(self.logfile):
                logtxt= "\t".join(["Server-Response:---name-sizebytes-t:", self.name, str(size), str(time.time()-self.start_time)])
                self.logfile.write(logtxt+"\n")
            return data
