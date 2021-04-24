#!/usr/bin/env python

from __future__ import print_function
import sys, os, glob, operator, time

if sys.version_info < (2, 7):
    sys.exit("ERROR: need python 2.7 or later for dep.py")

if __name__ == "__main__":
    log_file_name = sys.argv[1]
    log_file_dir = os.path.dirname(log_file_name)
    log_files = glob.glob(os.path.join(log_file_dir,"*.memlog"))
    build_mem_results = {}
    for logf in log_files:
        f = open(logf,'r')
        m = int(float(f.readline())/1024.)
        build_mem_results[os.path.basename(logf)[:-7]] = m
        f.close()
    f = open(log_file_name,'w')
    f.write("# (File Name, Peak RSS in MB)\n")
    first = True
    for it in sorted(build_mem_results.items(), key=operator.itemgetter(1),reverse=True):
        if first:
            first = False
            print("The top memory consumer is", it[0], it[1], "MB" )
        f.write(str(it)+'\n')
    f.close()
    print("More details are available at", sys.argv[1])
