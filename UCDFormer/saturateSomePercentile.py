import numpy as np

class saturateImage():
##Defines code for image adjusting/pre-processing

    def __init__(self):
        """Reset the class; indicates the class hasn't been initailized"""
        self.initialized = False

    def saturateSomePercentileMultispectral(self,inputMap,percentileToSaturate,axis):
        inputMap=inputMap.astype(float)
        inputMapNormalized=inputMap
        for iter in range(axis):
            inputMapBand=inputMap[iter, :, :]
            inputMapNormalizedBand=(inputMapBand-np.amin(inputMapBand))/(np.percentile(inputMapBand,(100-percentileToSaturate))-np.amin(inputMapBand)+np.exp(-10))
            inputMapNormalizedBand[inputMapNormalizedBand>1]=1
            inputMapNormalized[iter, :, :]=inputMapNormalizedBand
        return inputMapNormalized
    
