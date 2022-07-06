#!/usr/bin/env bash
sudo apt update
sudo apt install -y protobuf-compiler

. /etc/lsb-release
ubuntu_version="$DISTRIB_RELEASE"

if (( $(echo "$ubuntu_version == 20.04" |bc -l) )); then 
    echo "your system is 20.04, cp opendrive to vec lib now..."
    cp assets/package/rough_map_node_20 assets/package/rough_map_node
elif (( $(echo "$ubuntu_version == 18.04" |bc -l) )); then 
    echo "your system is 18.04, cp opendrive to vec lib now..."
    cp assets/package/rough_map_node_20 assets/package/rough_map_node
elif (( $(echo "$ubuntu_version == 16.04" |bc -l) )); then 
    echo "your system is 16.04, cp opendrive to vec lib now..."
    cp assets/package/rough_map_node_20 assets/package/rough_map_node
else
    echo "There is no version package for: $ubuntu_version"
fi
