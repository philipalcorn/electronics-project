import ltspice
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Load the .raw file
l = ltspice.Ltspice("Project Draft.raw")
l.parse()

# Extract frequency and complex data
freq = l.get_frequency()              # Frequency vector [Hz]
vout = l.get_data("V(vo)")           # Complex values (Re + jIm)

# Convert magnitude to dB
vout_db = 20 * np.log10(np.abs(vout))

# Optional: phase (degrees)
#vout_phase = np.angle(vout, deg=True)

# Prepare Features
x = np.log2(freq).reshape(-1,1)
y = vout_db

#Fit linear regression
model = LinearRegression().fit(x,y)
y_fit = model.predict(x)

slope = model.coef_[0]  # Slope in dB/oct
r2 = r2_score(y,y_fit)  # R Squared Value

# Finding max error
differences = y - y_fit
max_diff = np.max(np.abs(differences))

# Defining ideal line
# Setting the line to start at the first data point
x0 = x[0,0]
y0 = y[0]
ideal_line = y0 + (-3)*(x.flatten()-x0)




# Plot magnitude response
plt.semilogx(freq, vout_db, label="Measured")
#plt.semilogx(freq, ideal_line, "g:", label="Ideal Response")
plt.semilogx(freq, y_fit, "r:", label="Best Fit")
plt.xlabel("Frequency [Hz]")
plt.ylabel("Magnitude [dB]")
plt.grid(True, which="both")

plt.legend()

plt.text(0.5, 0.52,
         f"Slope = {slope:.2f} dB/oct\nR2 = {r2:.3f}\n"
         f"Max Diff= {max_diff:.2f} dB",
         transform=plt.gca().transAxes,
         color="black")

plt.show()

