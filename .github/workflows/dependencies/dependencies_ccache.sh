#!/usr/bin/env bash

set -eu -o pipefail

echo $PWD

wget https://github.com/ccache/ccache/releases/download/v$1/ccache-$1-linux-x86_64.tar.xz

tar xvf ccache-$1-linux-x86_64.tar.xz

sudo ln -s -f ccache-$1-linux-x86_64/ccache /usr/local/bin/

which ccache

#sudo apt-get update
#
#if [[ `lsb_release -r -s` == "20.04" ]]; then
#  sudo apt-get install -y --no-install-recommends libhiredis-dev libzstd-dev
#  wget https://github.com/ccache/ccache/releases/download/v4.5.1/ccache-4.5.1.tar.gz
#  tar xvfz ccache-4.5.1.tar.gz
#  cd ccache-4.5.1
#  mkdir build
#  cd build
#  cmake .. -DCMAKE_BUILD_TYPE=Release \
#           -DENABLE_DOCUMENTATION=OFF
#  make -j 2
#  sudo make install
#else
#  sudo apt-get install -y --no-install-recommends ccache
#fi
