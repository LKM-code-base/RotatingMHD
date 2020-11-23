import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import (AutoMinorLocator, MultipleLocator)

# Extracting time, velocity's x-component at P1, temperature at P1,
# Nusselt number at the right wall, pressure difference between P1 and
# P4, the average velocity metric and the average vorticity metric.
data = np.loadtxt("MIT_benchmark.txt", 
                  delimiter="|", 
                  usecols = (1,3,4,11,8,13,14),
                  skiprows = 1)

# Initializing averange and peak-to-valley values
results     = np.zeros(2 * len(data[0,:]) - 1)

# Reference values - the "true" solution
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

# The average and peak-to-valleys values are optained from the cycle
# that occurs between t = 1593 and t = 1598
search_point_1 = 159300 # t = 1593
search_point_2 = 159500 # t = 1595
search_point_3 = 159600 # t = 1596
search_point_4 = 159800 # t = 1598

for i in range(1, len(data[0,:])):
# Shifts to the local maximum and minima 
  shift_1 = np.argmin(data[search_point_1:search_point_2, i])
  shift_2 = np.argmax(data[search_point_1:search_point_3, i])
  shift_3 = np.argmin(data[search_point_3:search_point_4, i])

  # Average value between the two minima
  results[2 * i - 1]  = np.average(data[search_point_1 + shift_1 : 
                                        search_point_3 + shift_3, i])

  # Difference between the maximum and the first minimum
  results[2 * i]  = (data[search_point_1 + shift_2, i] - 
                     data[search_point_1 + shift_1, i])
  
  # The period is based on the temperature behaviour
  if i == 2:
    results[0] = (data[search_point_3 + shift_3, 0] -
                  data[search_point_1 + shift_1, 0])

for j in range(len(errors)):
  errors[j] = (results[j] - quere_data[j]) / quere_data[j] * 100.

composite_metrics[0] = 0.5 * (np.abs(errors[1]) + np.abs(errors[3]))
composite_metrics[1] = math.sqrt(0.5 * (errors[1]**2 + errors[3]**2))
composite_metrics[2] = 0.5 * (np.abs(errors[2]) + np.abs(errors[4]))
composite_metrics[3] = math.sqrt(0.5 * (errors[2]**2 + errors[4]**2))

accuracy_metrics[0] = (errors[1] + errors[3] + errors[5]) / 3.
accuracy_metrics[1] = (errors[0] + errors[2] + errors[4] + errors[6]) / 4.
accuracy_metrics[2] = (errors[0] + errors[1] + errors[2] + errors[3] +
                       errors[4] + errors[5] + errors[6]) / 7.

print(results)
print(errors)
print(composite_metrics)
print(accuracy_metrics)

#fig, ax = plt.subplots()
#ax.plot(data[:, 0], data[:, 5])

# Set axis ranges; by default this will put major ticks every 25.
#ax.set_xlim(1593, 1597.5)
#ax.set_ylim(0.24, 0.29)

# Change major ticks to show every 20.
#ax.xaxis.set_major_locator(MultipleLocator(1))
#ax.yaxis.set_major_locator(MultipleLocator(0.01))

# Change minor ticks to show every 5. (20/4 = 5)
#ax.xaxis.set_minor_locator(AutoMinorLocator(4))
#ax.yaxis.set_minor_locator(AutoMinorLocator(4))

# Turn grid on for both major and minor ticks and style minor slightly
# differently.
#ax.grid(which='major', color='#CCCCCC', linestyle=':')
#ax.grid(which='minor', color='#CCCCCC', linestyle=':')

#ax.set(xlabel='Simulation time', ylabel='Velocity')
#ax.grid()

#plt.show()