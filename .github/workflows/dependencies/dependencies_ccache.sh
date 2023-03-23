#!/usr/bin/env bash

set -eu -o pipefail

sudo apt-get update

sudo apt-get install -y --no-install-recommends ccache
