import os
import sys
import math
import json
import ROOT
import random

from PhysicsTools.NanoAODTools.postprocessing.framework.datamodel import Collection
from PhysicsTools.NanoAODTools.postprocessing.framework.eventloop import Module

from utils import deltaR,getCtauLabel

class TaggerWorkingpoints(Module):

    def __init__(
        self,
        inputCollection = lambda event: Collection(event, "Jet"),
        taggerName = "llpdnnx",
        outputName = "llpdnnx",
        predictionLabels = ["B","C","UDS","G","PU","isLLP_QMU_QQMU","isLLP_Q_QQ"], #this is how the output array from TF is interpreted
        logctauValues = range(-1, 4),
        multiplicities = [0, 1, 2],
        globalOptions={"isData":False}
    ):
        self.globalOptions = globalOptions
        self.inputCollection = inputCollection
        self.outputName = outputName
        self.predictionLabels = predictionLabels
        self.logctauValues = logctauValues
        self.multiplicities = multiplicities
        self.logctauLabels = map(lambda ctau: getCtauLabel(ctau),logctauValues)
        self.taggerName = taggerName
        
 
    def beginJob(self):
        pass
        
    def endJob(self):
        pass
        
    def beginFile(self, inputFile, outputFile, inputTree, wrappedOutputTree):
        self.out = wrappedOutputTree
        
        for ctau in self.logctauValues:
            for label in self.predictionLabels:
                for m in self.multiplicities:
                    self.out.branch(self.outputName+"_"+getCtauLabel(ctau)+"_"+label+"_min"+str(m),"F")
            if not self.globalOptions["isData"]:
                    
                for m in self.multiplicities:
                    self.out.branch(self.outputName+"_"+getCtauLabel(ctau)+"_"+label+"_min"+str(m),"F")

    def endFile(self, inputFile, outputFile, inputTree, wrappedOutputTree):
        pass
        
    def analyze(self, event):
        """process event, return True (go to next module) or False (fail, go to next event)"""
        jets = self.inputCollection(event)
        
        predictionsPerCtauAndClass = {ctau: {className: [] for className in self.predictionLabels} for ctau in self.logctauValues}
        for ijet,jet in enumerate(jets):
            if not hasattr(jet,self.taggerName):
                print "WARNING - jet ",jet," has no ",self.taggerName," result stored for ",self.outputName," -> skip"
                continue
            predictions = getattr(jet,self.taggerName)
            for ctau in self.logctauValues:
                for label in self.predictionLabels:
                    predictionsPerCtauAndClass[ctau][label].append(predictions[ctau][label])
                
        for ctau in self.logctauValues:
            for label in self.predictionLabels:
                predictionsPerCtauAndClass[ctau][label] = sorted(predictionsPerCtauAndClass[ctau][label],reverse=True)

                for m in self.multiplicities:
                    if m<len(predictionsPerCtauAndClass[ctau][label]):
                        self.out.fillBranch(self.outputName+"_"+getCtauLabel(ctau)+"_"+label+"_min"+str(m),predictionsPerCtauAndClass[ctau][label][m])
                    else:
                        self.out.fillBranch(self.outputName+"_"+getCtauLabel(ctau)+"_"+label+"_min"+str(m),0)
                        
                    
        return True
