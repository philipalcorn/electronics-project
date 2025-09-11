from configparser import RawConfigParser
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from PyLTSpice import SimRunner,SpiceEditor,RawRead
#from PyLTSpice.sim.process_callback import ProcessCallback
import time
import os
import shutil
import glob

import callbacks

class Spice:
    def __init__(self,ascPath,spicePath,outputFolder,callbackProc):
        self.ascPath=ascPath
        self.spicePath=spicePath

        # Automatically get .raw and .net paths
        #self.rawPath="."+(self.ascPath.split(".")[1])+"_1.raw"
        self.netPath="."+(self.ascPath.split(".")[1])+".net"

        # Create the simulator and netlist objects
        self.LTC=SimRunner(output_folder=outputFolder,parallel_sims=12)
        self.LTC.create_netlist(self.ascPath)
        self.netlist=SpiceEditor(self.netPath)

        self.callbackProc=callbackProc

    def simulate(self,runNetlistFile):
        # Delete old file to ensure we dont get old data
        #if os.path.exists(self.rawPath):
            #os.remove(self.rawPath)
        #if os.path.exists(self.netPath):
            #os.remove(self.netPath)
        #logPath="."+(self.rawPath.split(".")[1])+".log"
        #opRawPath="."+(self.rawPath.split(".")[1])+".op.raw"
        #if os.path.exists(logPath):
            #os.remove(logPath)
        #if os.path.exists(opRawPath):
            #os.remove(opRawPath)

        # Run LTSpice simulation
        self.LTC.run(self.netlist,run_filename=runNetlistFile,callback=self.callbackProc)

        # Make sure simulation has finished writing to .raw
        #self.waitRaw()
        #self.LTR=RawRead(self.rawPath)

    def waitRaw(self):
        # Ensure that the file exists before continuing
        start=time.time()
        while not os.path.exists(self.rawPath):
            if time.time()-start>20:
                raise TimeoutError("Could not open "+self.rawPath)

        # Make sure the data has finished writing
        size=os.path.getsize(self.rawPath)
        while size==0:
            size=os.path.getsize(self.rawPath)
        while os.path.getsize(self.rawPath)>size:
            size=os.path.getsize(self.rawPath)

    def getData(self,trace):
        if os.path.exists(self.rawPath):
            return self.LTR.get_trace(trace)
        else:
            raise FileNotFoundError("Could not find "+self.rawPath)

    def setVal(self,component,value):
        self.netlist.set_component_value(component,value)

class Optimizer:
    def __init__(self,spice,shelfs,cPath,rPath):
        self.spice=spice

        # Initalize common cap and resistor values as empty tuples
        self.cs=self.tupleFromFile(cPath)
        self.rs=self.tupleFromFile(rPath)

        # The first and last two iteration will be for R1, the feedback cap, and DC blocking cap
        self.shelfs=shelfs
        self.component=tuple(self.getComponentNames())

        # Get frequency data
        #self.spice.simulate()
        #self.freqx=np.array(self.spice.getData("frequency")).real.astype(float)
        #self.freqx=np.log2(self.freqx).flatten().reshape(-1,1) # flatten reshape may not be necessary


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
        for i in range(2,self.shelfs+2):
            components.append(f"R{i}")
            components.append(f"C{i}")
        return components

    def beginOptimization(self):
        #self.spice.setVal("R1",15e3)

        bestError=50 # arbitrary
        comp=self.component[0]
        for r in self.rs:
            print(f"Optimizing: {self.component[0]} {r}")
            self.spice.setVal(comp,r)
            error,vals=self.iteration(0,bestError,1e18)
            if error<self.bestError:
                bestError=error
                bestVals=[r]+vals
        
        print(bestVals)
        #set best components values and export

        vout=self.spice.getData("V(vo)")
        return vout,self.freq

    def iteration(self,n,errorBest,rc):
        comp1=self.component[n*2+1]
        comp2=self.component[n*2+2]
        bestError=errorBest

        for r in self.rs:
            print(f"Optimizing: {comp1} {r}")
            self.spice.setVal(comp1,r)
            for c in self.cs:
                print(f"Optimizing: {comp2} {c}")
                if r*c>rc:
                    continue
                self.spice.setVal(comp2,c)
                print(n,self.shelfs)
                if n!=self.shelfs-1:
                    error,vals=self.iteration(n+1,bestError,r*c)
                    if error<bestError:
                        bestError=error
                        bestVals=[r,c]+vals
                else:
                    self.spice.simulate()
                    for rawFile in self.spice.LTC:
                        error=self.calcError(rawFile)
                    if error<bestError:
                        bestError=error
                        bestVals=[r,c]
                    return bestError,bestVals

    def generateRaws(self):
        self.cleanBatchDir()
        scriptDir=os.path.dirname(os.path.abspath(__file__))
        batchDir=os.path.join(scriptDir,"batch")

        self.rs=tuple([1e3,5e3])
        #self.cs=tuple([30e-9,300e-9])

        #self.rs=tuple([1,10])
        self.cs=tuple([150e-9])

        #self.rs=tuple([10e3,6e3,2.5e3,900,350,10])
        #self.cs=tuple([470e-9,150e-9,68e-9,22e-9,15e-9])

        for r1 in self.rs:
            self.spice.netlist.set_component_value("R1",r1)
            for r2 in self.rs:
                self.spice.netlist.set_component_value("R2",r2)
                for c2 in self.cs:
                    self.spice.netlist.set_component_value("C2",c2)
                    r2c2=r2*c2
                    for r3 in self.rs:
                        self.spice.netlist.set_component_value("R3",r3)
                        for c3 in self.cs:
                            r3c3=r3*c3
                            if r3c3>r2c2:
                                continue
                            self.spice.netlist.set_component_value("C3",c3)
                            for r4 in self.rs:
                                self.spice.netlist.set_component_value("R4",r4)
                                for c4 in self.cs:
                                    r4c4=r4*c4
                                    if r4c4>r3c3:
                                        continue
                                    self.spice.netlist.set_component_value("C4",c4)
                                    for r5 in self.rs:
                                        self.spice.netlist.set_component_value("R5",r5)
                                        for c5 in self.cs:
                                            r5c5=r5*c5
                                            if r5c5>r4c4:
                                                continue
                                            self.spice.netlist.set_component_value("C5",c5)
                                            for r6 in self.rs:
                                                self.spice.netlist.set_component_value("R6",r6)
                                                for c6 in self.cs:
                                                    r6c6=r6*c6
                                                    if r6c6>r5c5:
                                                        continue
                                                    self.spice.netlist.set_component_value("C6",c6)
                                                    runNetlistFile="{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}".format((self.spice.netlist.netlist_file.name).split(".")[0],r1,r2,c2,r3,c3,r4,c4,r5,c5,r6,c6)
                                                    self.spice.simulate(runNetlistFile)
                                                    rawPath=os.path.join(batchDir,runNetlistFile+".raw")
                                                    #self.spice.LTC.wait_completion()
                                                    #if not os.path.exists(rawPath):
                                                        #print(f"Missing raw file {rawPath}")

        self.spice.LTC.wait_completion()
        print("Successful/Total Simulations: " + str(self.spice.LTC.okSim) + "/" + str(self.spice.LTC.runno))
        #self.deleteExtraFiles()

    def cleanBatchDir(self):
        scriptDir=os.path.dirname(os.path.abspath(__file__))
        batchDir=os.path.join(scriptDir,"batch")
        if os.path.isdir(batchDir):
            for item in os.listdir(batchDir):
                itemPath=os.path.join(batchDir,item)
                if os.path.isfile(itemPath) or os.path.islink(itemPath):
                    os.remove(itemPath)
                elif os.path.isdir(itemPath):
                    shutil.rmtree(itemPath)

    def deleteExtraFiles(self):
        scriptDir=os.path.dirname(os.path.abspath(__file__))
        batchDir=os.path.join(scriptDir,"batch")
        if os.path.isdir(batchDir):
            for item in os.listdir(batchDir):
                if item.endswith(".raw") and not item.endswith(".op.raw"):
                    continue
                if item.endswith(".op.raw") or item.endswith(".log") or item.endswith(".net"):
                    itemPath=os.path.join(batchDir,item)
                    try:
                        os.remove(itemPath)
                    except:
                        print(f"Error deleteing {itemPath}")
                    continue
                end=item.split("_")[-1]
                if end.isdigit():
                    itemPath=os.path.join(batchDir,item)
                    try:
                        os.remove(itemPath)
                    except:
                        print(f"Error deleteing {itemPath}")

    def findBest(self):
        freq=self.getFrequencyData()
        print("FREQ")
        print(freq)
        for vout in self.spice.LTC:
            print()
            print(vout)
            print(vout[500],freq[500])

    def getFrequencyData(self):
        spice=Spice(self.spice.ascPath,self.spice.spicePath,"./",callbacks.CallbackProcFreq)
        runNetlistFile="{}_{}".format((spice.netlist.netlist_file.name).split(".")[0],"Frequency")
        spice.simulate(runNetlistFile)

        #rawPath=runNetlistFile.split(".")[0]+".raw"
        #logPath=runNetlistFile.split(".")[0]+".log"
        #print(rawPath,logPath)
        #return callbacks.CallbackProcFreq.callback(rawPath,logPath)

        for freq in spice.LTC:
            return freq



    def calcError(self,rawFile):
        self.LTR=RawRead(rawFile)
        vout=self.spice.getData("V(vo)")

        # Convert magnitude to dB
        vout_db = 20 * np.log10(np.abs(vout))

        # Prepare Features
        y = vout_db

        #Fit linear regression
        model = LinearRegression().fit(self.freqx,y)
        y_fit = model.predict(self.freqx)

        slope = model.coef_[0]  # Slope in dB/oct
        m=10*np.log10(1/2)
        error=(slope/m-1)**2
        return error

#class CallbackProcVO(ProcessCallback):
#    @staticmethod
#    def callback(rawFile,_):
#        rawData=RawRead(rawFile)
#        #freq_trace = rawData.get_trace("frequency")
#        #freq = np.array(freq_trace.get_wave()).real.astype(float)
#        #freq = np.log2(freq).flatten()
#        vout_trace = rawData.get_trace("V(vo)")
#        vout = np.array(vout_trace.get_wave()).flatten()
#        return vout

#class CallbackProcFreq(ProcessCallback):
#    @staticmethod
#    def callback(rawFile,_):
#        rawData=RawRead(rawFile)
#        freq_trace = rawData.get_trace("frequency")
#        freq = np.array(freq_trace.get_wave()).real.astype(float)
#        freq = np.log2(freq).flatten()
#        return freq

        


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
    folder="./batch"
    callbackProc=callbacks.CallbackProcVO
    spice=Spice(ascPath,spicePath,folder,callbackProc)
    #spice.simulate()

    # Path to text file with common sizes
    rPath=r".\common_resistor_sizes.txt"
    cPath=r".\common_cap_sizes.txt"

    # Create optimizer object
    opt=Optimizer(spice,5,cPath,rPath)
    #vout,freq=opt.beginOptimization()
    opt.generateRaws()
    opt.findBest()
    return

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