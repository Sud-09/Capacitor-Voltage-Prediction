# Authors: Sudhanva Gokhale and Mackenzie Clark
# email: sudhanva@vt.edu and mjclark355@gmail.com

import numpy as np
import time
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# In the line below we will create data points for the first 2.1 secs of the capacitor charging
# keep in mind that the first entry in the 'arrange()' function has to be non zer
# here it is 0.1
generated_time_data = np.arange(0.1,2.1, 0.1)

# declaring the initial voltage and the final voltage of the capacitor
# if the initial voltage is not present just assign 0 to initial voltage
initial_voltage = 10 
final_voltage = 20 

# The time constant for the the capacitor will depend on your particular circuit 
time_const = 3 

# Generating voltage data for the first 2.1 secs and then adding realistic amount of 
# noise to the generated data. To change the noise level read online documentation for np.random 
ideal_voltage_data = initial_voltage + (final_voltage-initial_voltage) * (1 - np.exp(-generated_time_data/time_const))  # Use the second formulation from above
voltage_data = ideal_voltage_data + .01 + np.random.normal(scale=np.sqrt(np.max(ideal_voltage_data))/500, size=len(generated_time_data))  # Add noise

# The plot generated below gives us a visual idea of how much noise is added
ax = plt.axes()
ax.scatter(generated_time_data, voltage_data, label='Measured(noisy) data')
ax.plot(generated_time_data, ideal_voltage_data, 'k', label='Ideal Data')
ax.set_title('Noisy data vs Ideal Data')
ax.set_ylabel('voltage in volts')
ax.set_ylim(5, 25)
ax.set_xlabel('Time in secs')
ax.legend()
plt.show()

# In the few sections below we will compare the built-in curve_fit() function
# with our approach

# Below we will predict the final value of the voltage and the time constant of the capacitor
# using 'curve_fit()' function
popt, pcov = curve_fit(lambda t, final_voltage, time_const, initial_voltage: initial_voltage + (final_voltage - initial_voltage) * (1 - np.exp(-generated_time_data/time_const)), generated_time_data, voltage_data)
Final_VoltValue_curveFit = popt[0]
Tau_Value_curveFit = popt[1]

# Starting timer to keep track of how much time our approach takes
start_time = time.perf_counter_ns()
# subtracting initial voltage from each data point, as we already have the initial value 
# and do not need to predict it. (See 14674814-Regressions-et-equations-integrales.pdf to understand the mathematics better)
yarr = voltage_data-initial_voltage

# Creating an empty array for time 
S = np.array([0.0]*len(generated_time_data))
S[0]=0.0

# The code below is a direct representation of mathematics used in the research paper.
# (See 14674814-Regressions-et-equations-integrales.pdf to understand the mathematics better)
for n in range(1,len(generated_time_data)):
    S[n]=S[n-1]+.5*(yarr[n]+yarr[n-1])*(generated_time_data[n]-generated_time_data[n-1])

Arr1 = np.array([
    [sum(S**2),sum(S*generated_time_data),sum(S)],
    [sum(S*generated_time_data),sum(generated_time_data**2),sum(generated_time_data)],
    [sum(S),sum(generated_time_data),len(generated_time_data)]
    ])

Arr1 = np.linalg.inv(Arr1)

Arr2 = np.array([
    [sum(S*yarr)],
    [sum(generated_time_data*yarr)],
    [sum(yarr)]
    ])

Sol = np.matmul(Arr1,Arr2)

# these are the predicted values using our approach, but they are in an intermediate form
# see 14674814-Regressions-et-equations-integrales.pdf for details
A = Sol[0]
B = Sol[1]
C = Sol[2]

# converting the intermediate values to final form
FinalVolt_ourPred = -B/A
TimeConst_ourPred = A
# stoppint the timer
end_time = time.perf_counter_ns()

# creating data to plot curve with our predicted values
pred_time_vector= np.arange(0,25,0.1)
pred_time_vector = np.linspace(np.min(pred_time_vector), np.max(pred_time_vector), 20)
pred_voltage_value = initial_voltage+FinalVolt_ourPred*(1-np.exp(TimeConst_ourPred*pred_time_vector))

# Displaying the predicted values
print("Tau = {:0.3f}".format(float(-1/TimeConst_ourPred)))
print("Final Voltage = {:0.3f}".format(float(FinalVolt_ourPred+initial_voltage)), "V")
print("Eq. % Error at Vf = {:0.3f}".format(float(abs(((FinalVolt_ourPred+initial_voltage)-final_voltage)/(final_voltage))*100)), "%")
print("Total Execution Time for Approx. =", (end_time-start_time)/1000, "us")

n = 10

# Creating the curve using curve_fit() equation
time_curveFit = np.arange(0,25,0.1)
time_curveFit = np.linspace(np.min(time_curveFit), np.max(time_curveFit), 20)
Voltage_curveFit = initial_voltage + (Final_VoltValue_curveFit - initial_voltage) * (1 - np.exp(-time_curveFit/Tau_Value_curveFit))

# Creating the curve using ideal equations
x_ideal = np.arange(0,25,0.1)
x_ideal = np.linspace(np.min(x_ideal), np.max(x_ideal), 20)
y_ideal = initial_voltage + (final_voltage - initial_voltage) * (1 - np.exp(-time_curveFit/time_const))

# Plotting a comparative curve-plot
ax = plt.axes()
ax.scatter(generated_time_data, voltage_data, label='First 2.1 secs of data(with noise)')
ax.plot(x_ideal, y_ideal, 'r--', label='Mathematical eqn of cap charging')
ax.plot(time_curveFit, Voltage_curveFit, 'k', label='using Curve_fit()')
ax.plot(pred_time_vector, pred_voltage_value, 'b--', label='Using our approx.')
ax.set_title(r'Comparison of Ideal vs Calculated Fit')
ax.set_ylabel('Voltage in volts')
ax.set_ylim(5, 25)
ax.set_xlabel('Time in secs')
ax.legend()
ax.set_ylim(5)
plt.show()