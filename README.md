# Capacitor-Voltage-Prediction
A capacitor is a device which is used to store electric charge using a couple of conductors which are insulated from each other using an insulator. The waveform for the charging and discharging of the capacitor is non-linear. This is a characteristic property of the capacitor, but it can pose certain problems. If it takes a capacitor a certain amount of time to charge/discharge to a given voltage, for eg:- 5 secs, the speed of our circuit operation can be limited. As mentioned earlier, we would have to wait 5 secs to get the 'Final Voltage' value (given that the final voltage value is not a fixed value that we obtain at all times). To get around this problem we can use an algebraic 'Best Fit' approach to predict what the voltage will be after 5 secs with the data we have at 2 secs (can be a different time, depending on your application).
That being said, we know that the equation for capacitor discharging can be linearized easily logarithm and some rearrangement. On the other hand, the capacitor charging equation cannot be linearized just by using logarithms and rearrangement. This is where the proposed approach will come into use.
Summary :- 'To predict the capacitor voltage by using algebraic best fit approach we need an approximation that will help us solve non-linear charging and discharging equations'
A research paper and python code will be provided in this repository.
The "14674814-Regressions-et-equations-integrales" file in this repository provides the mathematical explanation of the process used to predict the final value.
The 'Cap_voltage_prediction.py' provides the python code for capacitor voltage prediction.

Created By Sudhanva Gokhale || sudhanva@vt.edu
       and Mackenzie Clark  || mjclark355@gmail.com 
