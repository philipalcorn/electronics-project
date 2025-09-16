import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from PyLTSpice import SimRunner,SpiceEditor
import time
import os
import callbacks

class Spice:
    def __init__(self,ascPath,spicePath,outputFolder,callbackProc):
        self.ascPath=ascPath
        self.spicePath=spicePath

        # Automatically get .net paths
        self.netPath="."+(self.ascPath.split(".")[1])+".net"

        # Create the simulator and netlist objects
        self.LTC=SimRunner(output_folder=outputFolder,parallel_sims=16)
        self.LTC.create_netlist(self.ascPath)
        self.netlist=SpiceEditor(self.netPath)

        # Define callback processor
        self.callbackProc=callbackProc

    def simulate(self,runNetlistFile):
        # Run LTSpice simulation
        self.LTC.run(self.netlist,run_filename=runNetlistFile,callback=self.callbackProc)

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
        #self.shelfs=shelfs
        #self.component=tuple(self.getComponentNames())

    @staticmethod
    def tupleFromFile(filepath):
        # Read the entire file content
        with open(filepath,'r') as file:
            text=file.read()

        # Split by commas and strip whitespace
        str_values=text.replace('\n',' ').split(',')
        values=tuple(float(val.strip()) for val in str_values if val.strip())

        return values

    def generateRaws(self):
        self.cleanBatchDir()
        scriptDir=os.path.dirname(os.path.abspath(__file__))
        batchDir=os.path.join(scriptDir,"batch")

        #self.rs=tuple([1e3,5e3])
        #self.cs=tuple([30e-9,300e-9])

        #self.rs=tuple([1,10])
        #self.cs=tuple([150e-9])
        self.rs=tuple([10e3,6e3,2.5e3,900,350,10])
        self.cs=tuple([470e-9,150e-9,68e-9,22e-9,15e-9])
        #self.spice.simulate(runNetlistFile="{}_{}".format((self.spice.netlist.netlist_file.name).split(".")[0],"default"))

        bestError=np.inf
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
                                                    runNetlistFile = "{}_{}".format((self.spice.netlist.netlist_file.name).split(".")[0],"_".join(f"{v:.0e}"for v in [r1,r2,c2,r3,c3,r4,c4,r5,c5,r6,c6]))
                                                    self.spice.simulate(runNetlistFile)
                                            self.spice.LTC.wait_completion()
                                            if "bestFile" not in locals():
                                                bestFile=runNetlistFile
                                            bestError,bestFile=self.findBest(bestError,bestFile)
                                            self.cleanBatchDir()
        print("Successful/Total Simulations: " + str(self.spice.LTC.okSim) + "/" + str(self.spice.LTC.runno))
        print("Optimization Finished")

    def cleanBatchDir(self):
        scriptDir=os.path.dirname(os.path.abspath(__file__))
        batchDir=os.path.join(scriptDir,"batch")
        if os.path.isdir(batchDir):
            for item in os.listdir(batchDir):
                itemPath=os.path.join(batchDir,item)
                os.remove(itemPath)

    def findBest(self,bestError,bestFile):
        freq=self.getFrequencyData()
        idealM=10*np.log10(1/2)
        logFreq=np.log2(freq)
        for vout,rawFile in self.spice.LTC:
            error=self.calcError(vout[0::10],freq[0::10],idealM,logFreq[0::10])
            #print(error)
            #error=self.calcError(vout,freq,idealM,logFreq)
            if error<bestError:
                bestError=error
                bestFile=rawFile
                print("New Best",bestError)
                print(self.components(bestFile))
        #print(bestFile,bestError)
        return bestError,bestFile
                

    def getFrequencyData(self):
        spice=Spice(self.spice.ascPath,self.spice.spicePath,"./",callbacks.CallbackProcFreq)
        runNetlistFile="{}_{}".format((spice.netlist.netlist_file.name).split(".")[0],"Frequency")
        spice.simulate(runNetlistFile)
        for freq in spice.LTC:
            return freq.reshape(-1,1)

    def calcError(self,vout,freq,idealM,logFreq):
        # Convert magnitude to dB
        voutDb=20*np.log10(np.abs(vout))
        model=LinearRegression().fit(logFreq,voutDb)
        yFit=model.predict(logFreq)
        #print(model.coef_[0])

        yMid=yFit[int(len(vout)/2)]
        xMid=logFreq[int(len(vout)/2)][0]

        idealVout=yMid+idealM*(logFreq.flatten()-xMid)
        error=np.linalg.norm(voutDb-idealVout,ord=2)
        return error

        # For debugging
        plt.semilogx(freq, voutDb, label="Measured")
        #plt.semilogx(freq, ideal_line, "g:", label="Ideal Response")
        plt.semilogx(freq, idealVout, "r:", label="Best Fit")
        plt.xlabel("Frequency [Hz]")
        plt.ylabel("Magnitude [dB]")
        plt.grid(True, which="both")

        plt.legend()
        plt.show()

        return error

    def components(self,file):
        parts=str(file).split("_")[2:]
        parts[-1]=parts[-1].split(".")[0]
        names=["R1","R2","C2", "R3","C3","R4","C4","R5","C5","R6","C6"]
        return dict(zip(names,parts))

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

    # Path to text file with common sizes
    rPath=r".\common_resistor_sizes.txt"
    cPath=r".\common_cap_sizes.txt"

    # Create optimizer object
    opt=Optimizer(spice,5,cPath,rPath)
    opt.generateRaws()
    opt.findBest()
    
if __name__=="__main__":
    main()