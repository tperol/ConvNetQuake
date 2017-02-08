# -------------------------------------------------------------------
# File Name : fig_comparison.py
# Creation Date : 08-12-2016
# Last Modified : Mon Jan 23 17:54:52 2017
# Author: Thibaut Perol <tperol@g.harvard.edu>
# -------------------------------------------------------------------
""" Script to generate the figures comparing the characteristics
of FAST and ConvNetQuake"""

import numpy as np
import matplotlib.pyplot as plt

def fig_memory_usage():

    # FAST memory
    x = [1,3,7,14,30,90,180]
    y_fast = [0.653,1.44,2.94,4.97,9.05,19.9,35.2]
    # ConvNetQuake
    y_convnet = [6.8*1e-5]*7
    # Create figure
    plt.loglog(x,y_fast,"o-")
    plt.hold('on')
    plt.loglog(x,y_convnet,"o-")
    # plot markers
    plt.loglog(x,[1e-5,1e-5,1e-5,1e-5,1e-5,1e-5,1e-5],'o')
    plt.ylabel("Memory usage (GB)")
    plt.xlabel("Continous data duration (days)")
    plt.xlim(1,180)
    plt.grid("on")
    plt.savefig("./figures/memoryusage.eps")
    plt.close()

def fig_run_time():
    # fast run time
    x_fast = [1,3,7,14,30,90,180]
    y_fast = [289,1.13*1e3,2.48*1e3,5.41*1e3,1.56*1e4,
              6.61*1e4,1.98*1e5]
    x_auto = [1,3]
    y_auto = [1.54*1e4, 8.06*1e5]
    x_convnet = [1,3,7,14,30]
    y_convnet = [9,27,61,144,291]
    # create figure
    plt.loglog(x_auto,y_auto,"o-")
    plt.hold('on')
    plt.loglog(x_fast[0:5],y_fast[0:5],"o-")
    plt.loglog(x_convnet,y_convnet,"o-")
    # plot x markers
    plt.loglog(x_convnet,[1e0]*len(x_convnet),'o')
    # plot y markers
    y_markers = [1,60,3600,3600*24]
    plt.plot([1]*4,y_markers,'ko')
    plt.ylabel("run time (s)")
    plt.xlabel("continous data duration (days)")
    plt.xlim(1,35)
    plt.grid("on")
    plt.savefig("./figures/runtimes.eps")



if __name__ == "__main__":
    fig_memory_usage()
    fig_run_time()
