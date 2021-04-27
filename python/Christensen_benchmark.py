import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import (AutoMinorLocator, MultipleLocator)

data = np.loadtxt("Christensen_Benchmark_case_0_5.txt",
                  delimiter="|",
                  usecols = (1, # Time
                             2, # Mean kinetic energy density
                             5, # Temperature at sample point
                             6, # Azimuthal velocity at sample point
                             8, # Drift frequency
                             4),# Sample point longitude
                  skiprows = 1)

for i in range(len(data[:,4])-1):
  data[i+1,4] = (data[i+1,5]-data[i,5])/(data[i+1,0]-data[i,0])

cwg_data  = np.array([58.3480,  # Mean average kinetic energy density
                      0.42812,  # Temperature at sample point
                      -10.1571, # Azimuthal velocity at sample point
                      0.1824]) # Drift frequency

errors    = np.zeros(len(cwg_data))
avg_data  = np.zeros(len(cwg_data))

fig, ax   = plt.subplots(4,1)

xmin      = 0.5
xmax      = 1.5

lwr_limit = np.where(data[:, 0] == xmin)[0][0]
upr_limit = np.where(data[:, 0] == xmax)[0][0]


for i in range(4):
  # Plot data
  ax[i].plot(data[lwr_limit:upr_limit, 0],
             data[lwr_limit:upr_limit, i+1])
  # Major axis every second
  avg_data[i] = np.average(data[lwr_limit:upr_limit, i+1])
  ax[i].xaxis.set_major_locator(MultipleLocator(1))
  # Minor axis every fourth of a second
  ax[i].xaxis.set_minor_locator(AutoMinorLocator(4))
  # Style and print grid
  ax[i].grid(which='major', color='#CCCCCC', linestyle='-')
  ax[i].grid(which='minor', color='#CCCCCC', linestyle=':')
  ax[i].grid(True)

# Output plot
plt.show()

for j in range(len(errors)):
  errors[j] = np.abs((avg_data[j] - cwg_data[j]) / cwg_data[j] * 100.)

print(" Averages \n", avg_data, "\n")
print(" Errors \n", errors, "\n")
