import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from PyLTSpice import SimRunner,SpiceEditor,RawRead
import time
import os

class Spice:
    def __init__(self,ascPath,spicePath):
        self.ascPath=ascPath
        self.spicePath=spicePath

        # Automatically get .raw and .net paths
        self.rawPath="."+(self.ascPath.split(".")[1])+"_1.raw"
        self.netPath="."+(self.ascPath.split(".")[1])+".net"

        # Create the simulator and netlist objects
        self.LTC=SimRunner()
        self.LTC.create_netlist(self.ascPath)
        self.netlist=SpiceEditor(self.netPath)
        self.LTR=RawRead(self.rawPath)

    def simulate(self):
        # Delete old file to ensure we dont get old data
        if os.path.exists(self.rawPath):
            os.remove(self.rawPath)

        # Run LTSpice simulation
        self.LTC.run(self.netlist)

        # Make sure simulation has finished writing to .raw
        self.waitRaw()

    def waitRaw(self):
        # Ensure that the file exists before continuing
        start=time.time()
        while not os.path.exists(self.rawPath):
            if time.time()-start>5:
                raise TimeoutError("Could not open "+self.rawPath)

        # Make sure the data has finished writing
        size=os.path.getsize(self.rawPath)
        print(size)
        if size==0:
            while size==0:
                size=os.path.getsize(self.rawPath)
        while os.path.getsize(self.rawPath)>size:
            print(size)
            size=os.path.getsize(self.rawPath)

    def getData(self,trace):
        if os.path.exists(self.rawPath):
            return self.LTR.get_trace(trace)
        else:
            raise FileNotFoundError("Could not find "+self.rawPath)

    def setVal(self,component,value):
        self.netlist.set_component_value(component,value)

class Optimizer:
    def __init__(self,shelfs,cPath,rPath):
        # Initalize common cap and resistor values as empty tuples
        self.cs=self.tupleFromFile(cPath)
        self.cs=self.tupleFromFile(rPath)

        # The first and last two iteration will be for R1, the feedback cap, and DC blocking cap
        self.iterations=shelfs*2+3
        self.component=tuple(self.getComponentNames())

    @staticmethod
    def tupleFromFile(filepath):
        # Read the entire file content
        with open(filepath,'r') as file:
            text=file.read()

        # Split by commas and strip whitespace
        str_values=text.replace('\n',' ').split(',')
        values=tuple(float(val.strip()) for val in str_values if val.strip())

        return values

    def getComponentNames(self):
        components=["R1"]
        shelfs=int((self.iterations-1)/2)
        for i in range(1,shelfs):
            components.append(f"R{i+1}")
            components.append(f"C{i+1}")
        components.append(f"C{int((self.iterations+1)/2)}")
        components.append(f"C{int((self.iterations+3)/2)}")
        return components

    def beginOptimization(self):
        self.iteration(0)
        print()

    def iteration(self,n):
        print(self.component[n],end=" ")
        if n==self.iterations-1:
            return
        else:
         self.iteration(n+1)

def infoOut(differences,slope,m,r2):
    # Prints various pieces of information
    p=2
    norm=np.linalg.norm(differences,ord=2)
    ideal="Ideal Slope"
    calculated="Calculated Slope"
    error="Calculated-Ideal"
    percent="Calculated/Ideal %"
    normStr=f"{p} Norm"
    r2Str="R^2"
    print(f"{ideal:^21}|{calculated:^21}|{error:^20}|{percent:^18}|{normStr:^13}|{r2Str:^20}")
    print(f"{m:.18f}|{slope:.18f}|{slope-m:.18f}|{slope/m*100:.15f}|{norm:.10f}|{r2:.18f}")

def graph(freq,vout_db,y_fit,ideal_line,slope,max_diff,r2):
    # Plot magnitude response
    plt.semilogx(freq, vout_db, label="Measured")
    plt.semilogx(freq, ideal_line, "g:", label="Ideal Response")
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

def main():
    # Create netlist for LTSpice
    ascPath=r".\Project_Draft.asc"
    spicePath=r"C:\Users\ruoom\AppData\Local\Programs\ADI\LTspice\LTspice.exe"
    spice=Spice(ascPath,spicePath)

    # Run LTSpice simulation
    spice.simulate()

    # Get data from LTSpice simulation
    vout=spice.getData("V(vo)")
    freq=np.array(spice.getData("frequency")).real.astype(float)

    # Path to text file with common sizes
    rPath=r".\common_resistor_sizes.txt"
    cPath=r".\common_cap_sizes.txt"

    # Create optimizer object
    opt=Optimizer(4,cPath,rPath)
    opt.beginOptimization()

    # Convert magnitude to dB
    vout_db = 20 * np.log10(np.abs(vout))

    # Prepare Features
    x = np.log2(freq).flatten()
    y = vout_db

    #Fit linear regression
    model = LinearRegression().fit(x.reshape(-1,1),y)
    y_fit = model.predict(x.reshape(-1,1))

    slope = model.coef_[0]  # Slope in dB/oct
    r2 = r2_score(y,y_fit)  # R Squared Value

    # Finding max error
    differences = y - y_fit
    max_diff = np.max(np.abs(differences))

    # Defining ideal line
    # Setting the line to start at the first data point
    x0 = x[0]
    y0 = y[0]
    m=10*np.log10(1/2)
    ideal_line = y0 + (m)*(x.flatten()-x0)

    infoOut(differences,slope,m,r2)

    graph(freq,vout_db,y_fit,ideal_line,slope,max_diff,r2)
    
if __name__=="__main__":
    main()