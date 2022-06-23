# ISWC2022Paper

## Supplemental material for ISWC 2022 paper "Distributed Processing of Property Path Queries in pure SPARQL".

### Repo Contents: 
* Virtual environment configuration file (Vagrantfile)
* Source code (all python code)
* Documentation of main code (including quick start instructions): RPQ_client_documentation.txt
* data from paper: directories dataBlue / dataGreen / dataRed, correspond to subgraphs colored white / gray / black (respectively) in paper -- data files inside are RDF n-triples format.
* additional data for local experiments: files graph_blue/green/red_3.txt contain the same data for experiments without using sparql (just processing RPQ on distributed graph data). The graphs in the "local" files are the same as in the RDF data above, they just use a custom format to work with legacy code.

### How to set up the virtual environment:
1. Install VirtualBox (free Oracle virtualization product, see virtualbox.org)
2. Install Vagrant (see vagrantup.com, command-line and configuration layer for virtual machines -- also free)
3. Copy all repo contents to a new directory
4. using the command line, from this directory enter 'vagrant up'
Note that the first setup of the virtual environment will be slow due to downloading of VM images + installing software (probably ~ 30 to 60 minutes depending on bandwidth). After that next time you access them it's very quick.
5. This creates a network of four virtual machines + your host machine. The virtual machines are all Ubuntu 18.04 machines running an Apache Jena Fuseki server each -- i.e., a SPARQL endpoint. As configured, the three SPARQL endpoints are reachable on your host machine at: http://localhost:3331/blue/sparql, http://localhost:3332/green/sparql, http://localhost:3333/red/sparql, and http://localhost:3334/yellow/sparql. The machines are fully and automatically configured from Vagrantfile.
6. To shutdown virtual machines, use command 'vagrant halt'
7. Note: the varantfile has been updated to use a box with java pre-installed, making first provisioning significantly faster.

### How to run tests (additional validation that does not appear in paper):

- Run script main_test.py with Python 3 (you may need to install dependencies (numpy, json, jinja2, urllib...). All these dependencies can be installed with command: pip3 install <package name>)
- What's happening? The Python code runs 12 single-source RPQ (property path queries) against the distributed data from four servers. The resutls are compared with those obtained from a RPQ on a centralized copy of the data. Currently only the number of responses is printed out to the terminal, but the lists of responses are checked. A log file for each query is also created, so that the communication (amount of data, time) is logged and can be used to create visualizations. The log also provides data to quantitatively compare query processing techniques.

### What more can you do?
- Add more RDF data to the servers (you can figure out how to provision it in the Vagrantfile, or else add it manually using the Fuseki web interface (accessible at http://localhost:3331/ ... 3334).
- Add more servers: see the configuration mechanism in the Vagrantfile: you can add another node to the network
- for all this and more, see RPQ_client_documentation.txt

### Notes: 
  * The paper indicates that the SPARQL endpoints on the network were Virtuoso servers. Unfortunately Virtuoso is buggy and our queries don't work on Virtuoso: we would need to rewrite them to get rid of all multi-source queries. Managing "differently abled" endpoints is future work.
  * This github account and repository are not 100% anonymized: don't go poking around if you don't want to find out who it belongs to (i.e. for blind review).


