#! /bin/bash
apt-get install -y cmake swig zlib1g-dev
conda create -n coach --file requirements.txt