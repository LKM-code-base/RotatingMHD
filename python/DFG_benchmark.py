import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import (AutoMinorLocator, MultipleLocator)
from scipy.signal import argrelextrema

data = np.loadtxt("dfg_benchmark_3.txt",
                  delimiter="|",
                  usecols = (2,   # Time
                             3,   # Pressure difference
                             4,   # Drag coefficient
                             5),  # Lift coefficient
                  skiprows = 1)

# Create plots
fig, ax = plt.subplots(3,1)

# These values are only valid with "dfg_benchmark_3.txt". They have to
# changed manually if other source files are used.
xmin      = 294.5
xmax      = 298

lwr_limit = np.where(data[:, 0] == xmin)[0][0]
upr_limit = np.where(data[:, 0] == xmax)[0][0]

# Index Quantity
# 0     Time
# 1     Velocity's x-component at P1
# 2     Temperature at P1
# 3     Nusselt number at the right wall
# 4     Pressure difference between P1 and P4
# 5     Average velocity metric
# 6     Average vorticity metric
for i in range(3):
  # Plot data
  ax[i].plot(data[lwr_limit:upr_limit, 0], data[lwr_limit:upr_limit, i+1])
  # Style and print grid
  ax[i].grid(which='major', color='#CCCCCC', linestyle='-')
  ax[i].grid(which='minor', color='#CCCCCC', linestyle=':')
  ax[i].grid(True)

# Output plot
#plt.show()

results     = np.zeros(4)
results[0]  = np.max(data[294607:297922,2])
results[1]  = np.max(data[294607:297922,3])
results[2]  = 1/(data[297922,0] - data[294607,0])
results[3]  = data[297922,1]

print(" Results \n", results, "\n")