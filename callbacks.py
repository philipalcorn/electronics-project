from PyLTSpice import RawRead
import numpy as np
from PyLTSpice.sim.process_callback import ProcessCallback

class CallbackProcVO(ProcessCallback):
    @staticmethod
    def callback(rawFile,_):
        rawData=RawRead(rawFile)
        #freq_trace = rawData.get_trace("frequency")
        #freq = np.array(freq_trace.get_wave()).real.astype(float)
        #freq = np.log2(freq).flatten()
        vout_trace = rawData.get_trace("V(vo)")
        vout = np.array(vout_trace.get_wave()).flatten()
        return vout,rawFile

class CallbackProcFreq(ProcessCallback):
    @staticmethod
    def callback(rawFile,_):
        rawData=RawRead(rawFile)
        freq_trace = rawData.get_trace("frequency")
        freq = np.array(freq_trace.get_wave()).real.astype(float).flatten()
        #freq = np.log2(freq)
        return freq