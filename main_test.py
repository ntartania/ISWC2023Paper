from RPQ_Client import Client, AsyncClient
from RPQ_Server_Proxy import Serveur, Server_proxy, Server_proxy_async
import matplotlib.pyplot as plt
import RPQ

def load_queries(qfile):
	with open(qfile) as qfile:
		allqueries = dict()
		cnt=0
		print ("loading queries")
		for line in qfile:
			cnt+=1
			#print("\n ---- SPARQL query ---------- ",cnt, "\n", line)
			start_node, regex = line.split() #"http://www.red.com/alibaba#p53"
			#regex= "(<casim:interaction>|<casim:interactions>|<casim:binding>|<casim:complex>|<casim:interacts>|<casim:interact>|<casim:complexes>)(<casim:interaction>|<casim:interactions>|<casim:binding>|<casim:complex>|<casim:interacts>|<casim:interact>|<casim:complexes>)*<casim:acetylation>(<casim:activation>|<casim:activity>|<casim:production>|<casim:induction>|<casim:overexpression>|<casim:up-regulation>|<casim:upregulation>|<casim:induces>|<casim:activates>|<casim:increases>)(<casim:activation>|<casim:activity>|<casim:production>|<casim:induction>|<casim:overexpression>|<casim:up-regulation>|<casim:upregulation>|<casim:induces>|<casim:activates>|<casim:increases>)*"
			regex=regex.replace("casim:", "http://wbi.informatik.hu-berlin.de/casim.rdfs#")
			allqueries[cnt]= (start_node, regex)
	return allqueries


def main():
	# load full graph
	fullgraph = RPQ.loadgraphNT('alibaba_full.nt')
	# ---- set up Server proxy objects
	sp1 = Server_proxy_async("serveur_blue",'http://localhost:3331/blue/sparql',"http://www.blue.com")
	sp2 = Server_proxy_async("serveur_green",'http://localhost:3332/green/sparql',"http://www.green.com")
	sp3 = Server_proxy_async("serveur_red",'http://localhost:3333/red/sparql',"http://www.red.com")
	sp4 = Server_proxy_async("serveur_yellow",'http://localhost:3334/yellow/sparql',"http://www.yellow.com")
	# ---- create and initialize querying client
	c2 = AsyncClient("client2")
	c2.initiate([sp1,sp2,sp3,sp4], debugging=False) #provide list of servers to query
	# set prefixes for sparql queries
	prefixes = {"casim": "<http://wbi.informatik.hu-berlin.de/casim.rdfs#>","blue": "<http://www.blue.com/alibaba#>","green": "<http://www.green.com/alibaba#>","red": "<http://www.red.com/alibaba#>","yellow": "<http://www.yellow.com/alibaba#>"}
	test_queries = load_queries("new_bio_queries_single_src.txt")
	for cnt in sorted(test_queries.keys()):
		print("\n ---- SPARQL query ---------- ",cnt)
		start_node, regex = test_queries[cnt] # each line contains start node + regular expression
		# log files for the communication with sparql endpoints (query time / response time / messages sizes are logged)
		log1 = "decomp_a_Q"+str(cnt)+".txt"
		results = c2.run_query_partial(regex, start_node=start_node, prefixes=prefixes, logfile=log1)
		print("== query answers [decomposed] ===========", len(results))#,"\n", "\t".join(sorted(results)))
		reference_results, a1, a2, a3 = RPQ.runquery(fullgraph, start_node, regex)
		print("== query answers [centralized] ===========", len(results))#,"\n", "\t".join(sorted(results)))
		if(set(reference_results)==set(results)):
			print("success: result lists match")
		else:
			print("test failed: result lists do not match:", results, reference_results)
		
	c2.close_all_sessions()

if __name__ == '__main__':
	main()