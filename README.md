# ISWC2022Paper

Supplemental material for ISWC 2022 paper "Distributed Processing of Property Path Queries in pure SPARQL".

_Note: this github account and repository are not 100% anonymous: don't go poking around if you don't want to find out who it belongs to._

Repo Contents: 
* Virtual environment configuration file (Vagrantfile)
* data from paper: directories dataBlue / dataGreen / dataRed, correspond to subgraphs colored white / gray / black (respectively) in paper
* Source code (all python code)

How to set up the virtual environment:
- Install VirtualBox (free Oracle virtualization product, see virtualbox.org)
- Install Vagrant (see vagrantup.com, command-line and configuration layer for virtual machines -- also free)
- Copy all repo contents to a new directory
- using the command line, from this directory enter 'vagrant up'
Note that the first setup of the virtual environment will be slow due to downloading of VM images + installing software (probably ~ 30 to 60 minutes depending on bandwidth). After that next time you access them it's very quick.
- This creates a network of three virtual machines + your host machine. The virtual machines are all Ubuntu 18.04 machines running an Apache Jena Fuseki server each -- i.e., a SPARQL endpoint. As configured, the three SPARQL endpoints are reachable on your host machine at: http://localhost:3331/blue/sparql, http://localhost:3332/green/sparql and http://localhost:3333/red/sparql. The machines are all configured in config file Vagrantfile
- To shutdown virtual machines, use command 'vagrant halt'

How to run tests (validation in paper):

- Run script testcode.py with Python3 (you may need to install dependencies (numpy, json, jinja2, urllib...). All these dependencies can be installed with command: pip3 install <package name>)
- What's happening? The Python code decomposes a Property path query (a single-source RPQ) and runs it over the distributed data from the three servers: two queries are sent to each server, the command-line output shows the SPARQL queries sent to each server, along with the responses from the servers. 
- Finally, it shows the results to the original RPQ: nodes http://www.blue.com/five and http://www.green.com/nine. This matches the expected answers, see data in paper, substituting colors blue/green/ref for white/gray/balck respectively.

What more can you do?
- Add more RDF data to the three servers (you can figure out how to provision it in the Vagrantfile, or else add it manually using the Fuseki web interface (accessible at http://localhost:3331/ ... 3333).
- Add more servers: see the configuration mechanism in the Vagrantfile: you can add another node to the network

Note: the paper indicates that the SPARQL endpoints on the network were Virtuoso servers. Unfortunately Virtuoso is buggy and our queries don't work on Virtuoso: we would need to rewrite them to get rid of all multi-source queries. Managing "differently abled" endpoints is future work.
