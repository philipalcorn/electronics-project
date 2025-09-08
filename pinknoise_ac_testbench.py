"""
Eric J. Wyers
wyers@tarleton.edu
ELEN 3445
Analog Pink Noise Generator Project

pinknoise_ac_testbench.py
"""

# this code comes with no warranty or guarantee of any kind

# this script does least-squares fitting on the psd of the pink noise 
# filter AC response;
# see pink_noise_shaping_filter_ac.asc
# or
# see pinknoise_ac_testbench.asc

import numpy as np
import matplotlib.pyplot as plt

def eng_str(x):
    from math import log10
    y = abs(x)
    if (y==0):
        z = 0
        engr_exponent = 0
        sign = ''
    else:
        exponent = int(log10(y)) 
        engr_exponent = exponent - exponent%3
        z = y/10**engr_exponent
        if (z>=0.1) and (z<1):
            z=z*1e3
            engr_exponent=engr_exponent-3
        sign = '-' if x < 0 else ''
    return sign+str(z)+'e'+str(engr_exponent)


# convenience parameters for saving plots and for selecting the file type
figsave = 0 #set to 1 to save figures; else 0
# png format may be more convenient for importing into your project report
figfiletype = 1 #set to 1 to save figures to pdf; else 0 for png

print('figsave:')
print(figsave)
print('figfiletype:')
print(figfiletype)

# set the value of p for the lp-norm psd error
# p = np.inf #max error
p = 2 #2-norm error
# p = 1 #1-norm error

# important: when saving ltspice ac data, in the "select traces to export"
# dialog, choose format cartesian (re,im) from the dropdown menu

# also important: be sure to use the following spice directives:
# .options plotwinsize=0 numdgt=15 measdgt=15
# .ac dec 1000 20 20k

# either use the modified .txt file with header removed:
# filename = 'pinknoise_ac_testbench_data_modified.txt' #header removed
# num_lines_to_remove = 0 #use this with the file with no header
# 
# or use the unmodified .txt file straight from LTSpice:
filename = 'responce.txt' #header not removed
num_lines_to_remove = 1 #use this with the file with header, header has 1 line

data = np.loadtxt(filename, skiprows=num_lines_to_remove, dtype=str)

# the freq array from the first column
freq = np.array(data[:,0].astype(float)).reshape(-1,1)
# the rectangular format values from the second column
v = np.array([complex(float(s.split(',')[0]),float(s.split(',')[1])) for s in data[:,1]]).reshape(-1,1)

vmagdb = 20*np.log10(np.abs(v))
vphase = (180/np.pi)*np.angle(v)

# Create a figure and a set of subplots
fig, (ax1, ax2) = plt.subplots(2, 1) # 2 rows, 1 column

# Plot on the first subplot
ax1.semilogx(freq, vmagdb, color='black')
ax1.set_title('LTSpice AC magnitude response')
ax1.set_ylabel('|Vout/Vin| [dB]')

# Plot on the second subplot
ax2.semilogx(freq, vphase, color='black')
ax2.set_title('LTSpice AC phase response')
ax2.set_xlabel('frequency [Hz]')
ax2.set_ylabel('<Vout/Vin [deg]')

# adjust layout to prevent overlapping titles/labels
plt.tight_layout()

if (figsave == 1):
    if (figfiletype == 1):
        # save the plot as a PDF
        plt.savefig("pinknoise_ac_testbench_plot1.pdf")
    else:
        # save the plot as a PNG file
        plt.savefig('pinknoise_ac_testbench_plot1.png')

# display the plot
plt.show()

# optional: close the figure to free up memory if you're done with it
# plt.close()

freqvec = freq
psdvec = vmagdb #psd = power spectral density

# now, set up the linear least-squares problem
Acol1 = np.log10(freqvec).reshape(-1,1)
Acol2 = np.ones((len(Acol1),1))
A = np.hstack((Acol1,Acol2))
d = psdvec.reshape(-1,1)
# now solve the Ax = d least-squares problem
xls = np.linalg.solve(A.T@A,A.T@d)

mls = xls[0]
bls = xls[1]

# this approach has too many moving parts:
# psdls = (mls*np.log10(freqvec)+bls).reshape(-1,1)
# this is the preferred way to do it:
psdls = A@xls.reshape(-1,1)

# now, extract the db/octave slope:
psd1 = psdls[0] 
psd2 = psdls[-1]
freq1 = freqvec[0] 
freq2 = freqvec[-1]
psd_delta = psd2-psd1
# number of octaves between f1 and f2 = log2(freq2/freq1)
freq_octaves = np.log10(freq2/freq1)/np.log10(2) #log2(.)=log10(.)/log10(2)
db_per_octave = psd_delta/freq_octaves

# compute error of pink noise slope relative to ideal:
# don't use this approximation of the ideal psd slope for pink noise:
# psd_slope_ideal = -3.0103
# instead, compute the exact ideal psd slope for pink noise below:
psd_slope_ideal = 10*np.log10(1/2)
psd_slope_error = np.abs(db_per_octave-psd_slope_ideal)

# compute the 2-norm error of the psd relative to the least-squares psd:
psd_lpnorm_error = np.linalg.norm(psdls-psdvec.reshape(-1,1),ord=p)

print('p value for lp-norm:')
print(p)

print('db/octave of pink noise data:')
print(eng_str(db_per_octave[0]))
print('ideal db/octave of pink noise:')
print(eng_str(psd_slope_ideal))
print('psd slope error:')
print(eng_str(psd_slope_error[0]))
print('lp-norm psd error:')
print(eng_str(psd_lpnorm_error))

# plot the PSD and the least-squares (LS) fit
plt.figure(figsize=(10, 6))
plt.semilogx(freqvec, psdvec)
plt.semilogx(freqvec, psdls, 'r--')
plt.title('PSD response of pink noise shaping filter (AC simulation)')
plt.xlabel('Frequency [Hz]')
plt.ylabel('Magnitude [dB]')
plt.legend(['data psd', 'LS fit psd'])
plt.grid(True)
if (figsave == 1):
    if (figfiletype == 1):
        # save the plot as a PDF
        plt.savefig("pinknoise_ac_testbench_plot2.pdf")
    else:
        # save the plot as a PNG file
        plt.savefig('pinknoise_ac_testbench_plot2.png')
plt.show()

# the psd slope error and lp-norm psd error for your designed pink noise
# shaping filter should be very similar to that listed below for my design;
# why? the results from this script are obtained with the AC testbench data, 
# and performance will only degrade once you start working in the simulated 
# time domain, and will degrade even more once you start taking measurements 
# of your physical pink noise generator circuit; also, this script only does 
# linear least-squares fitting, but the other scripts that I'll use to 
# assess your performance incorporate higher-order models, and so if your 
# design isn't as good as it can be at this stage, it will be more difficult 
# to achieve decent performance when using the other, more advanced
# assessment methods



# results for pinknoise_ac_testbench.asc:
# figsave:
# 0
# figfiletype:
# 1
#  p value for lp-norm:
# 2
# db/octave of pink noise data:
# -2.9891952507246784e0
# ideal db/octave of pink noise:
# -3.010299956639812e0
# psd slope error:
# 21.104705915133692e-3
# lp-norm psd error:
# 10.695922742479176e0
