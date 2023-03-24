#!/usr/bin/env bash

set -eu -o pipefail

wget https://github.com/ccache/ccache/releases/download/v${1}/ccache-${1}-linux-x86_64.tar.xz
tar xvf ccache-${1}-linux-x86_64.tar.xz
sudo cp -f ccache-${1}-linux-x86_64/ccache /usr/local/bin/
