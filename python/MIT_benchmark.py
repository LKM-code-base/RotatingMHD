import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import (AutoMinorLocator, MultipleLocator)
from scipy.signal import argrelextrema

# Extracting time, velocity's x-component at P1, temperature at P1,
# Nusselt number at the right wall, pressure difference between P1 and
# P4, the average velocity metric and the average vorticity metric.
# Note to the files: The number at the end of the file name
# ("MIT_benchmark_R*.txt") correspond to the refinement level.
data = np.loadtxt("MIT_benchmark_R3.txt",
                  delimiter="|",
                  usecols = (1,   # Time
                             2,   # Velocity x-component
                             4,   # Temperature
                             11,  # Nusselt number (left wall)
                             8,   # Pressure difference 14
                             13,  # Average velocity metric
                             14), # Average vorticity metric
                  skiprows = 1)

# Initializing averange and peak-to-valley values
results     = np.zeros(2 * len(data[0,:]) - 1)

# Reference values - the "true" solution (Taken from the MIT benchmark paper)
quere_data  = np.array([3.41150,                # Period
                        # Average,  Amplitude
                        0.056356,   0.054828,   # Velocity x-component
                        0.265480,   0.042740,   # Temperature
                        4.57946,    0.007100,   # Nusselt number
                        -0.001850,  0.020380])  # Pressure difference 14

# Initializing the errors
errors            = np.zeros(len(quere_data))

# Initializing composite metrics
composite_metrics = np.zeros(4)

# Initializing accuracy metrics
accuracy_metrics  = np.zeros(3)

# Create plots
fig, ax = plt.subplots(6,1)

xmin      = 1500
xmax      = 1600

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
for i in range(6):
  # Plot data
  ax[i].plot(data[lwr_limit:upr_limit, 0], data[lwr_limit:upr_limit, i+1])
  # Style and print grid
  ax[i].grid(which='major', color='#CCCCCC', linestyle='-')
  ax[i].grid(which='minor', color='#CCCCCC', linestyle=':')
  ax[i].grid(True)

# Output plot
#plt.show()

for i in range(1, len(data[0,:])):
  # Shifts to the local maximum and minima
  min_value = np.min(data[lwr_limit:upr_limit, i])
  max_value = np.max(data[lwr_limit:upr_limit, i])

  # Average value between the two minima
  results[2 * i - 1]  = np.average(data[lwr_limit:upr_limit, i])

  # Difference between the maximum and the first minimum
  results[2 * i]  = np.abs(max_value - min_value)


max_indices = argrelextrema(data[lwr_limit:upr_limit, 2], np.greater)
addition = 0.
for i in range(np.size(max_indices)-1):
  addition += data[lwr_limit:upr_limit, 0][max_indices[0][i+1]] - data[lwr_limit:upr_limit, 0][max_indices[0][i]]

results[0] = addition / (np.size(max_indices)-1)

# Compute relative errors
for j in range(len(errors)):
  errors[j] = np.abs((results[j] - quere_data[j]) / quere_data[j] * 100.)

# Compute composite metrics
composite_metrics[0] = 0.5 * (np.abs(errors[1]) + np.abs(errors[3]))
composite_metrics[1] = math.sqrt(0.5 * (errors[1]**2 + errors[3]**2))
composite_metrics[2] = 0.5 * (np.abs(errors[2]) + np.abs(errors[4]))
composite_metrics[3] = math.sqrt(0.5 * (errors[2]**2 + errors[4]**2))

# Compute accuracy metrics
accuracy_metrics[0] = (errors[1] + errors[3] + errors[5]) / 3.
accuracy_metrics[1] = (errors[0] + errors[2] + errors[4] + errors[6]) / 4.
accuracy_metrics[2] = (errors[0] + errors[1] + errors[2] + errors[3] +
                       errors[4] + errors[5] + errors[6]) / 7.

print(" Results \n", results, "\n")
print(" Le Quere's data \n", quere_data, "\n")
print(" Relative errors (w.r.t. Le Quere) \n", errors, "\n")
print(" Composite metrics \n", composite_metrics, "\n")
print(" Accuracy metrics \n", accuracy_metrics, "\n")