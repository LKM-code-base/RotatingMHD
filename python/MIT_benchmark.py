import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import (AutoMinorLocator, MultipleLocator)

# Extracting time, velocity's x-component at P1, temperature at P1,
# Nusselt number at the right wall, pressure difference between P1 and
# P4, the average velocity metric and the average vorticity metric.
# Note to the files: The number at the end of the file name 
# ("MIT_benchmark_*.txt") correspond to the refinement level.
# The second entry of the usecols list is for the 1st and 2nd 
# refinement level 3 and 2 respectively.
data = np.loadtxt("MIT_benchmark_2.txt", 
                  delimiter="|", 
                  usecols = (1,2,4,11,8,13,14),
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

# Initializing search points
search_point_1 = np.zeros(6, dtype=int)
search_point_2 = np.zeros(6, dtype=int)
search_point_3 = np.zeros(6, dtype=int)
search_point_4 = np.zeros(6, dtype=int)

# The search points are used to delimit ranges in which a 
# minimum or a maximum is. They are determined by visual evaluation of 
# the plots between t = 1592 and t = 1598 and are choosen as to be 
# compatible with both results
search_point_1[0] = 159300
search_point_1[1] = 159200
search_point_1[2] = 159300
search_point_1[3] = 159175
search_point_1[4] = 159300
search_point_1[5] = 159300

search_point_2[0] = 159400
search_point_2[1] = 159400
search_point_2[2] = 159450
search_point_2[3] = 159325
search_point_2[4] = 159500
search_point_2[5] = 159475

search_point_3[0] = 159600
search_point_3[1] = 159550
search_point_3[2] = 159625
search_point_3[3] = 159500
search_point_3[4] = 159625
search_point_3[5] = 159650

search_point_4[0] = 159800
search_point_4[1] = 159800
search_point_4[2] = 159800
search_point_4[3] = 159700
search_point_4[4] = 159850
search_point_4[5] = 159800

for i in range(1, len(data[0,:])):
  # Shifts to the local maximum and minima 
  shift_1 = np.argmin(data[search_point_1[i-1]:search_point_2[i-1], i])
  shift_2 = np.argmax(data[search_point_1[i-1]:search_point_3[i-1], i])
  shift_3 = np.argmin(data[search_point_3[i-1]:search_point_4[i-1], i])

  # Average value between the two minima
  results[2 * i - 1]  = np.average(data[search_point_1[i-1] + shift_1 : 
                                        search_point_3[i-1] + shift_3, i])

  # Difference between the maximum and the first minimum
  results[2 * i]  = (data[search_point_1[i-1] + shift_2, i] - 
                     data[search_point_1[i-1] + shift_1, i])
  
  # The period is based on the temperature behaviour
  if i == 2:
    results[0] = (data[search_point_3[i-1] + shift_3, 0] -
                  data[search_point_1[i-1] + shift_1, 0])

# Compute relative errors
for j in range(len(errors)):
  errors[j] = (results[j] - quere_data[j]) / quere_data[j] * 100.

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
print(" Relative errors (w.r.t. Le Quere) \n", errors, "\n")
print(" Composite metrics \n", composite_metrics, "\n")
print(" Accuracy metrics \n", accuracy_metrics, "\n")

# Create plots
fig, ax = plt.subplots(6,1)

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
  ax[i].plot(data[:, 0], data[:, i+1])
  # Visualize only between t = 1592 and t = 1598
  ax[i].set_xlim(1592, 1598)
  # Major axis every second
  ax[i].xaxis.set_major_locator(MultipleLocator(1))
  # Minor axis every fourth of a second
  ax[i].xaxis.set_minor_locator(AutoMinorLocator(4))
  # Style and print grid
  ax[i].grid(which='major', color='#CCCCCC', linestyle='-')
  ax[i].grid(which='minor', color='#CCCCCC', linestyle=':')
  ax[i].grid(True)

# Set axis ranges for each value. Depending on the refinement level
# the ranges may have to be adjusted
ax[0].set_ylim(0.03, 0.1)
ax[1].set_ylim(0.24, 0.29)
ax[2].set_ylim(4.575, 4.59)
ax[3].set_ylim(-1.5e-2, 1e-2)
ax[4].set_ylim(0.23949, 0.23955)
ax[5].set_ylim(3.015, 3.02)

# Output plot
plt.show()