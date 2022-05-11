from RPQ_Client import Client, Serveur, Server_proxy
# ---- Test case ----
print("\n ---- RPQ from data in files ---------- \n")
regex = "<ex:alpha><ex:bravo>*<ex:charlie><ex:delta>"
c1 = Client("client")
s1 = Serveur("serveur_blue", "http://www.blue.com", "graph_blue_3.txt")
s2 = Serveur("serveur_green", "http://www.green.com", "graph_green_3.txt")
s3 = Serveur("serveur_red", "http://www.red.com", "graph_red_3.txt")

graph_servers = [s1, s2, s3]

c1.initiate(graph_servers)

results = c1.run_single_source_query("http://www.blue.com/1", regex)
print("\n\n===== query results from files ===========\n", results)



# ---- Test case SPARQL query ----
print("\n ---- SPARQL query ---------- \n")
sp1 = Server_proxy("serveur_blue",'http://localhost:3331/blue/sparql',"http://www.blue.com")
sp2 = Server_proxy("serveur_green",'http://localhost:3332/green/sparql',"http://www.green.com")
sp3 = Server_proxy("serveur_red",'http://localhost:3333/red/sparql',"http://www.red.com")
c2 = Client("client2")
c2.initiate([sp1,sp2,sp3], logging=False) #provide list of servers to query
prefixes = {"ex": "<http://example.com/>"}
results = c2.run_single_source_query("http://www.blue.com/one", regex, prefixes)
print("\n == query results from servers =========== \n", results)