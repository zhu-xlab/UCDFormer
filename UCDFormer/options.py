"""
Code Author: Sudipan Saha.

"""

import torch
import argparse
import os



class optionsDCVA():
    """This class defines some options required for running DCVA
    """

    def __init__(self):
        """Reset the class; indicates the class hasn't been initailized"""
        self.initialized = False
        

    def initialize(self, parser):
        parser.add_argument('--dataPath', type=str, default='F:\\MCD_dataset\\1', help='path to the data. Data is assumed to be saved in a .mat file. with two variables - preChangeData and postChangeData')
        parser.add_argument('--inputChannels', type=str, default='RGB', help='input channels - can be RGB or RGBNIR')
        parser.add_argument('--layersToProcess', type=str, default='2,5,8', help='comma separated layers from which features are extracted, choose values from 2,5,8,10,11,23')
        parser.add_argument('--thresholding', type=str, default='adaptive', help='adaptive/otsu/scaledOtsu, adaptive - useful for complex urban areas. Otsu and Scaled Otsu for areas with less spatial complexity')
        parser.add_argument('--otsuScalingFactor', type=float, default=1.25, help='scaling factor is scaled otsu thresholding is used')
        parser.add_argument('--objectMinSize', type=int, default=128, help='minimum size for objects in pixel')
        parser.add_argument('--topPercentSaturationOfImageOk', type=bool, default=True, help='bool indicating if images are preprocessed to saturate high values')
        parser.add_argument('--topPercentToSaturate', type=float, default=1, help='percentage to saturate')
        ##Remaining options are multiple CD related. They can be ignored if only binary CD is performed.
        parser.add_argument('--multipleCDBool', type=bool, default=False, help='(True/False)whether multiple CD is needed. If false only binary CD will be performed')
        parser.add_argument('--changeVectorBinarizationStrategy', type=str, default='otsuThreshold', help='otsuThreshold/zeroThreshold, for multiple CD - strategy for binarizing deep change vector')
        parser.add_argument('--clusteringStrategy', type=str, default='hierarchical', help='hierarchical/kmodes, for multiple CD - strategy for clustering binarized deep change vector')
        parser.add_argument('--hierarchicalDistanceStrategy', type=str, default='hamming', help='hamming/correlation, for multiple CD hierarchical clusteting - method for calculating distance between binarized deep change vectors')
        parser.add_argument('--clusterNumber', type=int, default=2, help='for multiple CD - cluster number')
        
        
        self.initialized = True
        return parser

    

    def parseOptions(self):
        """Parse the options"""
        parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        parser = self.initialize(parser)
        opt = parser.parse_args()
        return opt
   