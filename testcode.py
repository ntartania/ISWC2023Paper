from RPQ_Client import Client, AsyncClient
from RPQ_Server_Proxy import Serveur, Server_proxy, Server_proxy_async
import matplotlib.pyplot as plt
import RPQ

def test_restrictions():
	t = RPQ.makeParseTree("<^apple>*<boat>*(<cat>|<^dog>)*")
	print("regex", t)
	nfa=t.to_DFA()
	print("DFA--------")
	nfa.uglyprint()
	nga2 = nfa.deepCopy()
	nga2.restrict_end(True)	
	print("restrict end to Forwards --------")
	nga2.uglyprint()
	print("regex=",nga2.to_regex())
	nga2 = nfa.deepCopy()
	nga2.restrict_end(False)	
	print("restrict end to Inverse --------")
	nga2.uglyprint()
	print("regex=",nga2.to_regex())
	nga2 = nfa.deepCopy()
	nga2.restrict_start(True)	
	print("restrict start to Forwards --------")
	nga2.uglyprint()
	print("regex=",nga2.to_regex())
	nga2 = nfa.deepCopy()
	nga2.restrict_start(False)	
	print("restrict start to Inverse --------")
	nga2.uglyprint()
	print("regex=",nga2.to_regex())



def test1():
	# ---- Test case ----
	print("\n ---- RPQ from data in files ---------- \n")
	regex = "<ex:alpha><ex:bravo>*<ex:charlie><ex:delta>"
	c1 = Client("client")
	s1 = Serveur("serveur_blue", "http://www.blue.com", "graph_blue_3.txt")
	s2 = Serveur("serveur_green", "http://www.green.com", "graph_green_3.txt")
	s3 = Serveur("serveur_red", "http://www.red.com", "graph_red_3.txt")
	
	graph_servers = [s1, s2, s3]
	
	c1.initiate(graph_servers)
	
	results = c1.run_query_partial(regex, start_node="http://www.blue.com/1")
	print("\n\n===== query results from files ===========\n", results)
	results = c1.run_query_partial(regex, end_node="http://www.blue.com/5")
	print("\n\n===== query results from files [reverse rpq] ===========\n", results)


def load_queries(qfile):
	with open(qfile) as qfile:
		allqueries = dict()
		cnt=0
		for line in qfile:
			cnt+=1
			print("\n ---- SPARQL query ---------- ",cnt, "\n", line)
			start_node, regex = line.split() #"http://www.red.com/alibaba#p53"
			#regex= "(<casim:interaction>|<casim:interactions>|<casim:binding>|<casim:complex>|<casim:interacts>|<casim:interact>|<casim:complexes>)(<casim:interaction>|<casim:interactions>|<casim:binding>|<casim:complex>|<casim:interacts>|<casim:interact>|<casim:complexes>)*<casim:acetylation>(<casim:activation>|<casim:activity>|<casim:production>|<casim:induction>|<casim:overexpression>|<casim:up-regulation>|<casim:upregulation>|<casim:induces>|<casim:activates>|<casim:increases>)(<casim:activation>|<casim:activity>|<casim:production>|<casim:induction>|<casim:overexpression>|<casim:up-regulation>|<casim:upregulation>|<casim:induces>|<casim:activates>|<casim:increases>)*"
			regex=regex.replace("casim:", "http://wbi.informatik.hu-berlin.de/casim.rdfs#")
			allqueries[cnt]= (start_node, regex)
	return allqueries

def test2():
	# ---- Test case SPARQL query ----
	print("\n ---- SPARQL query ---------- \n")
	regex = "<ex:alpha><ex:bravo>*(<^ex:echo>|<ex:charlie>)"
	regex=regex.replace("ex:", "http://example.com/")
	
	sp1 = Server_proxy("serveur_blue",'http://localhost:3331/blue/sparql',"http://www.blue.com")
	sp2 = Server_proxy("serveur_green",'http://localhost:3332/green/sparql',"http://www.green.com")
	sp3 = Server_proxy("serveur_red",'http://localhost:3333/red/sparql',"http://www.red.com")
	c2 = Client("client2")
	c2.initiate([sp1,sp2,sp3], debugging=True) #provide list of servers to query
	prefixes = {"ex": "<http://example.com/>"}
	#c2.test_sparql_queries("http://www.blue.com/1", regex, prefixes=prefixes, realstep1=True)
	#results = c2.run_single_source_query("http://www.blue.com/one", regex, prefixes)
	results = c2.run_query_partial(regex, start_node="http://www.blue.com/one")
	print("\n\n===== query results from 1 ===========\n", results)
	results = c2.run_query_partial(regex, end_node="http://www.green.com/seven")
	#results, edgelist = c2.iterative_strategy_1sq("http://www.blue.com/one", regex, True, prefixes)
	print("\n == query results to 7 =========== \n", results)

def test3(realstep1=True, realstep2=True):
	# ---- Test case SPARQL query ----
	#print("\n ---- SPARQL query ---------- \n")
	sp1 = Server_proxy("serveur_blue",'http://localhost:3331/blue/sparql',"http://www.blue.com")
	sp2 = Server_proxy("serveur_green",'http://localhost:3332/green/sparql',"http://www.green.com")
	sp3 = Server_proxy("serveur_red",'http://localhost:3333/red/sparql',"http://www.red.com")
	sp4 = Server_proxy("serveur_yellow",'http://localhost:3334/yellow/sparql',"http://www.yellow.com")
	c2 = Client("client2")
	c2.initiate([sp1,sp2,sp3,sp4], debugging=False) #provide list of servers to query, Logging is debug mode
	prefixes = {"casim": "<http://wbi.informatik.hu-berlin.de/casim.rdfs#>","blue": "<http://www.blue.com/alibaba#>","green": "<http://www.green.com/alibaba#>","red": "<http://www.red.com/alibaba#>","yellow": "<http://www.yellow.com/alibaba#>"}
	start_node = "http://www.red.com/alibaba#p53"
	regex= "(<casim:interaction>|<casim:interactions>|<casim:binding>|<casim:complex>|<casim:interacts>|<casim:interact>|<casim:complexes>)(<casim:interaction>|<casim:interactions>|<casim:binding>|<casim:complex>|<casim:interacts>|<casim:interact>|<casim:complexes>)*<casim:acetylation>(<casim:activation>|<casim:activity>|<casim:production>|<casim:induction>|<casim:overexpression>|<casim:up-regulation>|<casim:upregulation>|<casim:induces>|<casim:activates>|<casim:increases>)(<casim:activation>|<casim:activity>|<casim:production>|<casim:induction>|<casim:overexpression>|<casim:up-regulation>|<casim:upregulation>|<casim:induces>|<casim:activates>|<casim:increases>)*"
	regex=regex.replace("casim:", "http://wbi.informatik.hu-berlin.de/casim.rdfs#")
	if(realstep1 and realstep2):
		results, edgelist = c2.iterative_strategy_1sq(start_node, regex,  prefixes=prefixes, logfile="0iterlog.txt")
		nodes = set()
		edges = set()
		with open("data_graph2.debug.txt", "w") as logf:
			for edge in edgelist:
				n1, e, n2 = edge
				nodes.add(n1)
				nodes.add(n2)
				if (not(edge in edges)):
					edges.add(edge)
				logf.write("\t".join([n1, e, n2])+"\n")
		print("data graph:", len(nodes), "nodes,", len(edges), "edges")
		print("== query answers [iterative] ===========", len(results),"\n", "\t".join(results))

		results = c2.run_query_partial(regex, start_node=start_node, one_query=True, prefixes=prefixes, logfile="0decomplog.txt")
		print("== query answers [decomposed] ===========", len(results),"\n", "\t".join(results))

	else : #do real step1 or none
		print("\n == test 3 dry run == \n")
		c2.test_sparql_queries(start_node, regex, prefixes=prefixes, realstep1=realstep1)



def test3A():
	# ---- Test case SPARQL query ----
	#print("\n ---- SPARQL query ---------- \n")
	sp1 = Server_proxy_async("serveur_blue",'http://localhost:3331/blue/sparql',"http://www.blue.com")
	sp2 = Server_proxy_async("serveur_green",'http://localhost:3332/green/sparql',"http://www.green.com")
	sp3 = Server_proxy_async("serveur_red",'http://localhost:3333/red/sparql',"http://www.red.com")
	sp4 = Server_proxy_async("serveur_yellow",'http://localhost:3334/yellow/sparql',"http://www.yellow.com")
	c2 = AsyncClient("client2")
	c2.initiate([sp1,sp2,sp3,sp4], debugging=False) #provide list of servers to query, Logging is debug mode
	prefixes = {"casim": "<http://wbi.informatik.hu-berlin.de/casim.rdfs#>","blue": "<http://www.blue.com/alibaba#>","green": "<http://www.green.com/alibaba#>","red": "<http://www.red.com/alibaba#>","yellow": "<http://www.yellow.com/alibaba#>"}
	start_node = "http://www.red.com/alibaba#p53"
	regex= "(<casim:interaction>|<casim:interactions>|<casim:binding>|<casim:complex>|<casim:interacts>|<casim:interact>|<casim:complexes>)(<casim:interaction>|<casim:interactions>|<casim:binding>|<casim:complex>|<casim:interacts>|<casim:interact>|<casim:complexes>)*<casim:acetylation>(<casim:activation>|<casim:activity>|<casim:production>|<casim:induction>|<casim:overexpression>|<casim:up-regulation>|<casim:upregulation>|<casim:induces>|<casim:activates>|<casim:increases>)(<casim:activation>|<casim:activity>|<casim:production>|<casim:induction>|<casim:overexpression>|<casim:up-regulation>|<casim:upregulation>|<casim:induces>|<casim:activates>|<casim:increases>)*"
	regex=regex.replace("casim:", "http://wbi.informatik.hu-berlin.de/casim.rdfs#")
	results = c2.run_query_partial(regex, start_node=start_node, one_query=True, prefixes=prefixes, logfile="0decomplog.txt")
	print("== query answers [decomposed] ===========", len(results))#,"\n", "\t".join(results))
	results, edgelist = c2.iterative_strategy_1sq(start_node, regex,  prefixes=prefixes, logfile="0iterlog.txt")
	nodes = set()
	edges = set()
	with open("data_graph2.debug.txt", "w") as logf:
		for edge in edgelist:
			n1, e, n2 = edge
			nodes.add(n1)
			nodes.add(n2)
			if (not(edge in edges)):
				edges.add(edge)
#			logf.write("\t".join([n1, e, n2])+"\n")
	print("data graph:", len(nodes), "nodes,", len(edges), "edges")
	print("== query answers [iterative 2] ===========", len(results))#,"\n", "\t".join(results))

	c2.close_all_sessions()


def test3Rev():
	# ---- Test case SPARQL query ----
	#print("\n ---- SPARQL query ---------- \n")
	sp1 = Server_proxy("serveur_blue",'http://localhost:3331/blue/sparql',"http://www.blue.com")
	sp2 = Server_proxy("serveur_green",'http://localhost:3332/green/sparql',"http://www.green.com")
	sp3 = Server_proxy("serveur_red",'http://localhost:3333/red/sparql',"http://www.red.com")
	sp4 = Server_proxy("serveur_yellow",'http://localhost:3334/yellow/sparql',"http://www.yellow.com")
	c2 = Client("client2")
	c2.initiate([sp1,sp2,sp3,sp4], debugging=False) #provide list of servers to query, Logging is debug mode
	prefixes = {"casim": "<http://wbi.informatik.hu-berlin.de/casim.rdfs#>","blue": "<http://www.blue.com/alibaba#>","green": "<http://www.green.com/alibaba#>","red": "<http://www.red.com/alibaba#>","yellow": "<http://www.yellow.com/alibaba#>"}
	end_node = "http://www.red.com/alibaba#Daxx"
	regex= "(<casim:interaction>|<casim:interactions>|<casim:binding>|<casim:complex>|<casim:interacts>|<casim:interact>|<casim:complexes>)(<casim:interaction>|<casim:interactions>|<casim:binding>|<casim:complex>|<casim:interacts>|<casim:interact>|<casim:complexes>)*<casim:acetylation>(<casim:activation>|<casim:activity>|<casim:production>|<casim:induction>|<casim:overexpression>|<casim:up-regulation>|<casim:upregulation>|<casim:induces>|<casim:activates>|<casim:increases>)(<casim:activation>|<casim:activity>|<casim:production>|<casim:induction>|<casim:overexpression>|<casim:up-regulation>|<casim:upregulation>|<casim:induces>|<casim:activates>|<casim:increases>)*"
	regex=regex.replace("casim:", "http://wbi.informatik.hu-berlin.de/casim.rdfs#")
	results = c2.run_query_partial(regex, end_node=end_node, one_query=True, prefixes=prefixes, logfile="0decomplog.txt")
	print("== query answers [decomposed / reverse RPQ] ===========", len(results),"\n", "\t".join(results))



def test3B(): #nulti-source!
	# ---- Test case SPARQL query ----
	#print("\n ---- SPARQL query ---------- \n")
	sp1 = Server_proxy_async("serveur_blue",'http://localhost:3331/blue/sparql',"http://www.blue.com")
	sp2 = Server_proxy_async("serveur_green",'http://localhost:3332/green/sparql',"http://www.green.com")
	sp3 = Server_proxy_async("serveur_red",'http://localhost:3333/red/sparql',"http://www.red.com")
	sp4 = Server_proxy_async("serveur_yellow",'http://localhost:3334/yellow/sparql',"http://www.yellow.com")
	c2 = AsyncClient("client2")
	c2.initiate([sp1,sp2,sp3,sp4], debugging=False) #provide list of servers to query, Logging is debug mode
	prefixes = {"casim": "<http://wbi.informatik.hu-berlin.de/casim.rdfs#>","blue": "<http://www.blue.com/alibaba#>","green": "<http://www.green.com/alibaba#>","red": "<http://www.red.com/alibaba#>","yellow": "<http://www.yellow.com/alibaba#>"}
	start_node = "http://www.red.com/alibaba#p53"
	regex= "(<casim:interaction>|<casim:interactions>|<casim:binding>|<casim:complex>|<casim:interacts>|<casim:interact>|<casim:complexes>)(<casim:interaction>|<casim:interactions>|<casim:binding>|<casim:complex>|<casim:interacts>|<casim:interact>|<casim:complexes>)*<casim:acetylation>(<casim:activation>|<casim:activity>|<casim:production>|<casim:induction>|<casim:overexpression>|<casim:up-regulation>|<casim:upregulation>|<casim:induces>|<casim:activates>|<casim:increases>)(<casim:activation>|<casim:activity>|<casim:production>|<casim:induction>|<casim:overexpression>|<casim:up-regulation>|<casim:upregulation>|<casim:induces>|<casim:activates>|<casim:increases>)*"
	regex=regex.replace("casim:", "http://wbi.informatik.hu-berlin.de/casim.rdfs#")
	results = c2.run_query_partial(regex, start_node=None, one_query=True, prefixes=prefixes, logfile="0decomplog.txt")
	print("== query answers [decomposed] ===========", len(results))#,"\n", "\t".join(results))

	c2.close_all_sessions()


def test30():
	# ---- Test case SPARQL query ----
	#print("\n ---- SPARQL query ---------- \n")
	print("\n ---- RPQ from data in files ---------- \n")
	c1 = Client("client")
	s1 = Serveur("serveur_blue", "http://www.blue.com", "dataBlue/alibaba.nt")
	s2 = Serveur("serveur_green", "http://www.green.com", "dataGreen/alibaba.nt")
	s3 = Serveur("serveur_red", "http://www.red.com", "dataRed/alibaba.nt")
	s4 = Serveur("serveur_yellow", "http://www.yellow.com", "dataYellow/alibaba.nt")
	
	graph_servers = [s1, s2, s3, s4]
	
	c1.initiate(graph_servers)
	
	prefixes = {"casim": "<http://wbi.informatik.hu-berlin.de/casim.rdfs#>","blue": "<http://www.blue.com/alibaba#>","green": "<http://www.green.com/alibaba#>","red": "<http://www.red.com/alibaba#>","yellow": "<http://www.yellow.com/alibaba#>"}
	start_node = "http://www.red.com/alibaba#p53"
	regex= "(<casim:interaction>|<casim:interactions>|<casim:binding>|<casim:complex>|<casim:interacts>|<casim:interact>|<casim:complexes>)(<casim:interaction>|<casim:interactions>|<casim:binding>|<casim:complex>|<casim:interacts>|<casim:interact>|<casim:complexes>)*<casim:acetylation>(<casim:activation>|<casim:activity>|<casim:production>|<casim:induction>|<casim:overexpression>|<casim:up-regulation>|<casim:upregulation>|<casim:induces>|<casim:activates>|<casim:increases>)(<casim:activation>|<casim:activity>|<casim:production>|<casim:induction>|<casim:overexpression>|<casim:up-regulation>|<casim:upregulation>|<casim:induces>|<casim:activates>|<casim:increases>)*"
	regex=regex.replace("casim:", "http://wbi.informatik.hu-berlin.de/casim.rdfs#")
	results = c1.run_query_partial(regex, start_node=start_node)
	print("\n\n===== query results from files ===========\n", len(results))
	
	# if(realstep1 and realstep2):
	# 	results, edgelist = c2.iterative_strategy_1sq(start_node, regex,  prefixes=prefixes, logfile="0iterlog.txt")
	# 	nodes = set()
	# 	edges = set()
	# 	with open("data_graph2.debug.txt", "w") as logf:
	# 		for edge in edgelist:
	# 			n1, e, n2 = edge
	# 			nodes.add(n1)
	# 			nodes.add(n2)
	# 			if (not(edge in edges)):
	# 				edges.add(edge)
	# 			logf.write("\t".join([n1, e, n2])+"\n")
	# 	print("data graph:", len(nodes), "nodes,", len(edges), "edges")
	# 	print("== query answers [iterative] ===========", len(results),"\n", "\t".join(results))

	# 	results = c2.run_query_partial(start_node, regex, one_query=True, prefixes=prefixes, logfile="0decomplog.txt")
	# 	print("== query answers [decomposed] ===========", len(results),"\n", "\t".join(results))

	# else : #do real step1 or none
	# 	print("\n == test 3 dry run == \n")
	# 	c2.test_sparql_queries(start_node, regex, prefixes=prefixes, realstep1=realstep1)


def test4():
	# ---- Test case SPARQL query ----
	sp1 = Server_proxy("serveur_blue",'http://localhost:3331/blue/sparql',"http://www.blue.com")
	sp2 = Server_proxy("serveur_green",'http://localhost:3332/green/sparql',"http://www.green.com")
	sp3 = Server_proxy("serveur_red",'http://localhost:3333/red/sparql',"http://www.red.com")
	sp4 = Server_proxy("serveur_yellow",'http://localhost:3334/yellow/sparql',"http://www.yellow.com")
	c2 = Client("client2")
	c2.initiate([sp1,sp2,sp3,sp4], debugging=False) #provide list of servers to query
	prefixes = {"casim": "<http://wbi.informatik.hu-berlin.de/casim.rdfs#>","blue": "<http://www.blue.com/alibaba#>","green": "<http://www.green.com/alibaba#>","red": "<http://www.red.com/alibaba#>","yellow": "<http://www.yellow.com/alibaba#>"}
	cnt =0
	test_queries = load_queries("new_bio_queries_single_src.txt")
	for cnt in sorted(test_queries.keys()):
		print("\n ---- SPARQL query ---------- ",cnt)
		start_node, regex = test_queries[cnt]
		log1 = "decomp_Q"+str(cnt)+".txt"
		log2 = "iter_Q"+str(cnt)+".txt"
		results = c2.run_query_partial(regex, start_node=start_node, prefixes=prefixes, logfile=log1)
		print("== query answers [decomposed] ===========", len(results))#,"\n", "\t".join(sorted(results)))
		results, edgelist = c2.iterative_strategy_1sq(start_node, regex, prefixes=prefixes, logfile=log2)

		print("== query answers [iterative] ===========", len(results))#,"\n", "\t".join(sorted(results)))
	

def test4A():
	# ---- Test case SPARQL query ----
	sp1 = Server_proxy_async("serveur_blue",'http://localhost:3331/blue/sparql',"http://www.blue.com")
	sp2 = Server_proxy_async("serveur_green",'http://localhost:3332/green/sparql',"http://www.green.com")
	sp3 = Server_proxy_async("serveur_red",'http://localhost:3333/red/sparql',"http://www.red.com")
	sp4 = Server_proxy_async("serveur_yellow",'http://localhost:3334/yellow/sparql',"http://www.yellow.com")
	c2 = AsyncClient("client2")
	c2.initiate([sp1,sp2,sp3,sp4], debugging=False) #provide list of servers to query
	prefixes = {"casim": "<http://wbi.informatik.hu-berlin.de/casim.rdfs#>","blue": "<http://www.blue.com/alibaba#>","green": "<http://www.green.com/alibaba#>","red": "<http://www.red.com/alibaba#>","yellow": "<http://www.yellow.com/alibaba#>"}
	cnt =0
	test_queries = load_queries("new_bio_queries_single_src.txt")
	for cnt in sorted(test_queries.keys()):
		print("\n ---- SPARQL query ---------- ",cnt)
		start_node, regex = test_queries[cnt]
		log1 = "decomp_a_Q"+str(cnt)+".txt"
		log2 = "iter_a_Q"+str(cnt)+".txt"
		results = c2.run_query_partial(regex, start_node=start_node, prefixes=prefixes, logfile=log1)
		print("== query answers [decomposed] ===========", len(results))#,"\n", "\t".join(sorted(results)))
		results, edgelist = c2.iterative_strategy_1sq(start_node, regex, prefixes=prefixes, logfile=log2)	
		nodes, edges  =set(), set()
		for edge in edgelist:
			n1, e, n2 = edge
			nodes.add(n1)
			nodes.add(n2)
			if (not(edge in edges)):
				edges.add(edge)
		print("data graph:", len(nodes), "nodes,", len(edges), "edges")
		print("== query answers [iterative] ===========", len(results))#,"\n", "\t".join(sorted(results)))
	c2.close_all_sessions()


def test4B():
	# ---- Test case SPARQL query ----
	sp1 = Server_proxy_async("serveur_blue",'http://localhost:3331/blue/sparql',"http://www.blue.com")
	sp2 = Server_proxy_async("serveur_green",'http://localhost:3332/green/sparql',"http://www.green.com")
	sp3 = Server_proxy_async("serveur_red",'http://localhost:3333/red/sparql',"http://www.red.com")
	sp4 = Server_proxy_async("serveur_yellow",'http://localhost:3334/yellow/sparql',"http://www.yellow.com")
	c2 = AsyncClient("client2")
	c2.initiate([sp1,sp2,sp3,sp4], debugging=False) #provide list of servers to query
	prefixes = {"casim": "<http://wbi.informatik.hu-berlin.de/casim.rdfs#>","blue": "<http://www.blue.com/alibaba#>","green": "<http://www.green.com/alibaba#>","red": "<http://www.red.com/alibaba#>","yellow": "<http://www.yellow.com/alibaba#>"}
	cnt =0
	test_queries = load_queries("new_bio_queries_single_src.txt")
	for cnt in sorted(test_queries.keys()):
		print("\n ---- SPARQL query [multi-source] ---------- ",cnt)
		start_node, regex = test_queries[cnt]
		log1 = "decomp_ms_Q"+str(cnt)+".txt"
		results = c2.run_query_partial(regex, start_node=None, prefixes=prefixes, logfile=log1)
		print("== query answers [decomposed] ===========", len(results))#,"\n", "\t".join(sorted(results)))

	c2.close_all_sessions()

def test5():
	c2 = Client("client2")
	#prefixes = {"casim": "<http://wbi.informatik.hu-berlin.de/casim.rdfs#>","blue": "<http://www.blue.com/alibaba#>","green": "<http://www.green.com/alibaba#>","red": "<http://www.red.com/alibaba#>","yellow": "<http://www.yellow.com/alibaba#>"}
	regex= "(<casim:interaction>|<casim:interactions>|<casim:binding>|<casim:complex>|<casim:interacts>|<casim:interact>|<casim:complexes>)(<casim:interaction>|<casim:interactions>|<casim:binding>|<casim:complex>|<casim:interacts>|<casim:interact>|<casim:complexes>)*<casim:acetylation>(<casim:activation>|<casim:activity>|<casim:production>|<casim:induction>|<casim:overexpression>|<casim:up-regulation>|<casim:upregulation>|<casim:induces>|<casim:activates>|<casim:increases>)(<casim:activation>|<casim:activity>|<casim:production>|<casim:induction>|<casim:overexpression>|<casim:up-regulation>|<casim:upregulation>|<casim:induces>|<casim:activates>|<casim:increases>)*"
	#regex=regex.replace("casim:", "http://wbi.informatik.hu-berlin.de/casim.rdfs#")
	c2.initiate([], debugging=False) #provide list of servers to query
	c2.test_NFA(regex)

def parse_results(fname):
	with open(fname) as resfile:
		data_in = [0]
		data_out = [0]
		data_total = [0]
		tpoints = [0]
		for line in resfile:
			data = line.split()
			if (not (line.startswith("Query") or line.startswith("Server"))):
				break
			server_name, size, t = data[1], int(data[2]), float(data[3])
			tpoints.append(t)
			if line.startswith("Query"):
				data_in.append(0)
				data_out.append(size)
			else:
				data_out.append(0)
				data_in.append(size)
			data_total.append(data_total[-1]+size) #cumulative total data exchanged

		return tpoints, data_in, data_out, data_total

def make_plots(plot_fname):
	tp1, din1, dout1, data1 = parse_results('0iterlog.txt')
	tp2, din2, dout2, data2 = parse_results('0decomplog.txt')
	fig, ax = plt.subplots()
	ax.plot(tp1, data1, 'b+')
	ax.plot(tp2, data2, 'r+')

	ax.set(xlabel='time (s)', ylabel='data sent/received (bytes)', title='Cumulative data over time')
		#ax.grid()

	fig.savefig(plot_fname)
	plt.show()

def make_12_plots(plot_fname):
	fig, ax = plt.subplots(4, 3, figsize=(8,5))
	fig.tight_layout()

	for i in range(4):
		for j in range(3):
			qnum = i*3+j+1 #query number from 1 to 12
			print("query:", qnum)
			tp1, din1, dout1, data1 = parse_results('iter_a_Q'+str(qnum)+'.txt')
			tp2, din2, dout2, data2 = parse_results('decomp_a_Q'+str(qnum)+'.txt')
			data1= [d1/1000 for d1 in data1]
			data2= [d2/1000 for d2 in data2]
			ax[i, j].plot(tp2, data2, 'ro')
			ax[i, j].plot(tp1, data1, 'b+')
			

			ax[i, j].set(xlabel='time (s)', ylabel='data sent/received (kb)', title='Query '+str(qnum))
		#ax.grid()

	fig.savefig(plot_fname)
	plt.show()


if __name__ == '__main__':
	#test3Rev()
	#test_restrictions()
	test2()
	#make_plots("fig1.png")
	#make_12_plots("compare_a_Q1-12.png")