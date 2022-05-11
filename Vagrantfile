# -*- mode: ruby -*-
# vi: set ft=ruby :

# virtuoso configuration and loading data
$script = <<-SCRIPT
if [ -d "apache-jena-fuseki-4.5.0" ] ; then
  echo "Already have fuseki."
else
  echo "Getting Fusecki"
  wget https://archive.apache.org/dist/jena/binaries/apache-jena-fuseki-4.5.0.tar.gz
  tar xvfz apache-jena-fuseki-4.5.0.tar.gz
  wget https://archive.apache.org/dist/jena/binaries/apache-jena-4.5.0.tar.gz
  tar xvf apache-jena-4.5.0.tar.gz
fi

SCRIPT

$script1 = <<-SCRIPT
mkdir mydb
apache-jena-4.5.0/bin/tdb2.tdbloader --loc /home/vagrant/mydb /home/vagrant/data/blue.nt
SCRIPT

$exec1 = <<-SCRIPT
cd apache-jena-fuseki-4.5.0/
java -jar fuseki-server.jar --tdb2 --loc=/home/vagrant/mydb /blue &
SCRIPT

$script2 = <<-SCRIPT
mkdir mydb
apache-jena-4.5.0/bin/tdb2.tdbloader --loc /home/vagrant/mydb /home/vagrant/data/green.nt
SCRIPT

$exec2 = <<-SCRIPT
cd apache-jena-fuseki-4.5.0/
java -jar fuseki-server.jar --tdb2 --loc=/home/vagrant/mydb /green &
SCRIPT

$script3 = <<-SCRIPT
mkdir mydb
apache-jena-4.5.0/bin/tdb2.tdbloader --loc /home/vagrant/mydb /home/vagrant/data/red.nt
SCRIPT

$exec3 = <<-SCRIPT
cd apache-jena-fuseki-4.5.0/
java -jar fuseki-server.jar --tdb2 --loc=/home/vagrant/mydb /red &
SCRIPT

# All Vagrant configuration is done below. The "2" in Vagrant.configure
# configures the configuration version (we support older styles for
# backwards compatibility). Please don't change it unless you know what
# you're doing.
Vagrant.configure("2") do |config|
  # The most common configuration options are documented and commented below.
  # For a complete reference, please see the online documentation at
  # https://docs.vagrantup.com.


  # Every Vagrant development environment requires a box. You can search for
  # boxes at https://vagrantcloud.com/search.
  config.vm.define "ServerBlue2" do |config|
    config.vm.box = "paanini/ubuntu1804-openjdk-11"
    config.vm.hostname = "Virtual11"
    config.vm.network "private_network", ip: "192.168.56.20"
    config.vm.network "forwarded_port", guest: 80, host: 8881
    config.vm.network "forwarded_port", guest: 8080, host: 8891
    config.vm.network "forwarded_port", guest: 3030, host: 3331
    config.vm.synced_folder "./dataBlue", "/home/vagrant/data"
    config.vm.provision "shell", inline: $script
    config.vm.provision "shell", inline: $script1
    config.vm.provision "shell", inline: $exec1, run: "always"

  end

  config.vm.define "ServerGreen2" do |config|
    config.vm.box = "ubuntu/bionic64"
    config.vm.hostname = "Virtual21"
    config.vm.network "private_network", ip: "192.168.56.21"
    config.vm.network "forwarded_port", guest: 80, host: 8882
    config.vm.network "forwarded_port", guest: 8080, host: 8892
    config.vm.network "forwarded_port", guest: 3030, host: 3332
    config.vm.synced_folder "./dataGreen", "/home/vagrant/data"
    config.vm.provision "shell", inline: $script
    config.vm.provision "shell", inline: $script2
    config.vm.provision "shell", inline: $exec2, run: "always"
  end

  config.vm.define "ServerRed2" do |config|
    config.vm.box = "ubuntu/bionic64"
    config.vm.hostname = "Virtual31"
    config.vm.network "private_network", ip: "192.168.56.22"
    config.vm.network "forwarded_port", guest: 80, host: 8883
    config.vm.network "forwarded_port", guest: 8080, host: 8893
    config.vm.network "forwarded_port", guest: 3030, host: 3333
    config.vm.synced_folder "./dataRed", "/home/vagrant/data"
    config.vm.provision "shell", inline: $script
    config.vm.provision "shell", inline: $script3
    config.vm.provision "shell", inline: $exec3, run: "always"
  end
  # Disable automatic box update checking. If you disable this, then
  # boxes will only be checked for updates when the user runs
  # `vagrant box outdated`. This is not recommended.
  # config.vm.box_check_update = false

  # Create a forwarded port mapping which allows access to a specific port
  # within the machine from a port on the host machine. In the example below,
  # accessing "localhost:8080" will access port 80 on the guest machine.
  # NOTE: This will enable public access to the opened port
  # config.vm.network "forwarded_port", guest: 80, host: 8080

  # Create a forwarded port mapping which allows access to a specific port
  # within the machine from a port on the host machine and only allow access
  # via 127.0.0.1 to disable public access
  # config.vm.network "forwarded_port", guest: 80, host: 8080, host_ip: "127.0.0.1"

  # Create a private network, which allows host-only access to the machine
  # using a specific IP.
  # config.vm.network "private_network", ip: "192.168.33.10"

  # Create a public network, which generally matched to bridged network.
  # Bridged networks make the machine appear as another physical device on
  # your network.
  # config.vm.network "public_network"

  # Share an additional folder to the guest VM. The first argument is
  # the path on the host to the actual folder. The second argument is
  # the path on the guest to mount the folder. And the optional third
  # argument is a set of non-required options.
  # config.vm.synced_folder "../data", "/vagrant_data"

  # Provider-specific configuration so you can fine-tune various
  # backing providers for Vagrant. These expose provider-specific options.
  # Example for VirtualBox:
  #
  # config.vm.provider "virtualbox" do |vb|
  #   # Display the VirtualBox GUI when booting the machine
  #   vb.gui = true
  #
  #   # Customize the amount of memory on the VM:
  #   vb.memory = "1024"
  # end
  #
  # View the documentation for the provider you are using for more
  # information on available options.

  # Enable provisioning with a shell script. Additional provisioners such as
  # Ansible, Chef, Docker, Puppet and Salt are also available. Please see the
  # documentation for more information about their specific syntax and use.
  # config.vm.provision "shell", inline: <<-SHELL
  #   apt-get update
  #   apt-get install -y apache2
  # SHELL
end
