import os
import sys
import math
import argparse
import random
import ROOT
import json
import numpy as np

from PhysicsTools.NanoAODTools.postprocessing.framework.postprocessor \
    import PostProcessor
from PhysicsTools.NanoAODTools.postprocessing.framework.datamodel \
    import Collection, Object
from PhysicsTools.NanoAODTools.postprocessing.framework.eventloop import Module
from PhysicsTools.NanoAODTools.modules import *

parser = argparse.ArgumentParser()

parser.add_argument('--isData', dest='isData',
                    action='store_true', default=False)
parser.add_argument('--isSignal', dest='isSignal',
                    action='store_true', default=False)
parser.add_argument('--year', dest='year',
                    action='store', type=int, default=2018)
parser.add_argument('--noiso', dest='noiso',
                    action='store_true', default=False)
parser.add_argument('--notrigger', dest='notrigger',
                    action='store_true', default=False)
parser.add_argument('--notagger', dest='notagger',
                    action='store_true', default=False)
parser.add_argument('--nobdt', dest='nobdt',
                    action='store_true', default=False)
parser.add_argument('--overwrite_pu', action='store', default=None)
parser.add_argument('--leptons', dest='leptons', type=int, default=2, choices=[1,2])
parser.add_argument('--nLeptons', dest='nLeptons', type=int, default=5)
parser.add_argument('--input', dest='inputFiles', action='append', default=[])
parser.add_argument('--cutflow', dest='cutflow', action='store_true', default=False)

parser.add_argument('output', nargs=1)

args = parser.parse_args()

print "isData:",args.isData
print "inputs:",len(args.inputFiles)
isSignal = False


for inputFile in args.inputFiles:
    if args.isSignal or "dirac" in inputFile or "majorana" in inputFile or "LLPGun" in inputFile: 
        isSignal = True
    if args.year<0 and ("-2016" in inputFile or "Run2016" in inputFile):
        year = 2016
    elif args.year<0 and ("-2017" in inputFile or "Run2017" in inputFile):
        year = 2017
    elif args.year<0 and ("-2018" in inputFile or "Run2018" in inputFile):
        year = 2018
    else:
        year = args.year
    rootFile = ROOT.TFile.Open(inputFile)
    if not rootFile:
        print "CRITICAL - file '"+inputFile+"' not found!"
        sys.exit(1)
    tree = rootFile.Get("Events")
    if not tree:
        print "CRITICAL - 'Events' tree not found in file '"+inputFile+"'!"
        sys.exit(1)
    print " - ", inputFile, ", events=", tree.GetEntries()

puProcessName = args.overwrite_pu

print "year:", year
print "isSignal:",isSignal
print "apply lepton iso: ","True" if args.noiso is True else "False (default)"
print "apply trigger selection: ","True" if args.notrigger is True else "False (default)"
print "run BDT: ","True" if args.nobdt is True else "False (default)"
print "run tagger: ","True" if args.notagger is True else "False (default)"
print "channel: ","single lepton" if args.leptons==1 else "dilepton"
print "output directory:", args.output[0]

globalOptions = {
    "isData": args.isData,
    "isSignal": isSignal,
    "year": year
}

isMC = not args.isData

minMuonPt = {2016: 26., 2017: 29., 2018: 26.}
minElectronPt = {2016: 29., 2017: 34., 2018: 34.}

if isMC:
    jecTags = {2016: 'Summer16_07Aug2017_V11_MC',
               2017: 'Fall17_17Nov2017_V32_MC',
               2018: 'Autumn18_V19_MC'
               }

    jerTags = {2016: 'Summer16_25nsV1_MC',
               2017: 'Fall17_V3_MC',
               2018: 'Autumn18_V7_MC'
               }

if args.isData:
    jecTags = {2016: 'Summer16_07Aug2017All_V11_DATA',
               2017: 'Fall17_17Nov2017_V32_DATA',
               2018: 'Autumn18_V19_DATA'
               }

met_variable = {
        2016: "MET",
        2017: "METFixEE2017",
        2018: "MET",
        }


leptonSelection = [
    MuonSelection(
        outputName="LooseMuons",
    ),
    MuonSelection(
        inputCollection=lambda event: event.LooseMuons,
        outputName="TriggeringMuons",
        muonMinPt=9.,
        muonMinDxysig=6.,
    ),
    EventSkim(selection=lambda event: event.nLooseMuons > 0, outputName="Muons"),
    EventSkim(selection=lambda event: event.nTriggeringMuons > 0, outputName="TrgMuon"),
    SingleMuonTriggerSelection(
        inputCollection=lambda event: event.TriggeringMuons,
    ),
    LeptonCollecting(
        tightMuonsCollection = lambda event: [],
        tightElectronsCollection = lambda event: [],
        looseMuonCollection = lambda event: event.LooseMuons,
        looseElectronCollection = lambda event: [],
        outputName = "Leptons",
    ),
]

analyzerChain = []

analyzerChain.extend(leptonSelection)

if args.notrigger is False:
    trigger_matched = lambda event: any([muon.isTriggerMatched>0 for muon in event.TriggeringMuons])
    analyzerChain.extend([
        EventSkim(selection=lambda event: (event.DisplacedMuonTrigger_flag) > 0,  outputName="l1_trigger"),
        EventSkim(selection=trigger_matched, outputName="l1_triggermatch"),
    ])
    
analyzerChain.append(
    EventSkim(selection=lambda event: event.nLooseMuons>=args.nLeptons, outputName="MinMuons")
)

featureDictFile = "${CMSSW_BASE}/src/PhysicsTools/NanoAODTools/data/nn/201117/experimental_feature_dict.py"


taggerModelPath = {
    2016: "${CMSSW_BASE}/src/PhysicsTools/NanoAODTools/data/nn/201117/weightMixed2016_ExtNominalNetwork_photon_DA_300_wasserstein4_lr001_201117.pb",
    2017: "${CMSSW_BASE}/src/PhysicsTools/NanoAODTools/data/nn/201117/weightMixed2017_ExtNominalNetwork_photon_DA_300_wasserstein4_lr001_201117.pb",
    2018: "${CMSSW_BASE}/src/PhysicsTools/NanoAODTools/data/nn/201117/weightMixed2018_ExtNominalNetwork_photon_DA_300_wasserstein4_lr001_201117.pb"
}

BDT2lmodelPathExperimental = {
    2016: "${CMSSW_BASE}/src/PhysicsTools/NanoAODTools/data/bdt/201117/experimental/bdt_2016.model",
    2017: "${CMSSW_BASE}/src/PhysicsTools/NanoAODTools/data/bdt/201117/experimental/bdt_2017.model",
    2018: "${CMSSW_BASE}/src/PhysicsTools/NanoAODTools/data/bdt/201117/experimental/bdt_2018.model"
}

BDT2lmodelPathUncorr = {
    2016: "${CMSSW_BASE}/src/PhysicsTools/NanoAODTools/data/bdt/201117/uncorrelated/bdt_2l_2016.model",
    2017: "${CMSSW_BASE}/src/PhysicsTools/NanoAODTools/data/bdt/201117/uncorrelated/bdt_2l_2017.model",
    2018: "${CMSSW_BASE}/src/PhysicsTools/NanoAODTools/data/bdt/201117/uncorrelated/bdt_2l_2018.model"
}

BDT1lmodelPathUncorr = {
    2016: "${CMSSW_BASE}/src/PhysicsTools/NanoAODTools/data/bdt/201117/uncorrelated/bdt_1l_lr0.100_min0.1000_depths4_bagging0.75_2016.model",
    2017: "${CMSSW_BASE}/src/PhysicsTools/NanoAODTools/data/bdt/201117/uncorrelated/bdt_1l_lr0.100_min0.1000_depths4_bagging0.75_2017.model",
    2018: "${CMSSW_BASE}/src/PhysicsTools/NanoAODTools/data/bdt/201117/uncorrelated/bdt_1l_lr0.100_min0.1000_depths4_bagging0.75_2018.model",
}

jesUncertaintyFile = {
    2016: "${CMSSW_BASE}/src/PhysicsTools/NanoAODTools/data/jme/Summer16_07Aug2017_V11_MC_Uncertainty_AK4PFchs.txt",
    2017: "${CMSSW_BASE}/src/PhysicsTools/NanoAODTools/data/jme/Fall17_17Nov2017_V32_MC_Uncertainty_AK4PFchs.txt",
    2018: "${CMSSW_BASE}/src/PhysicsTools/NanoAODTools/data/jme/Autumn18_V19_MC_Uncertainty_AK4PFchs.txt"
}
jerResolutionFile = {
    2016: "${CMSSW_BASE}/src/PhysicsTools/NanoAODTools/data/jme/Summer16_25nsV1_MC_PtResolution_AK4PFchs.txt",
    2017: "${CMSSW_BASE}/src/PhysicsTools/NanoAODTools/data/jme/Fall17_V3_MC_PtResolution_AK4PFchs.txt",
    2018: "${CMSSW_BASE}/src/PhysicsTools/NanoAODTools/data/jme/Autumn18_V7_MC_PtResolution_AK4PFchs.txt"
}

jerSFUncertaintyFile = {
    2016: "${CMSSW_BASE}/src/PhysicsTools/NanoAODTools/data/jme/Summer16_25nsV1_MC_SF_AK4PFchs.txt",
    2017: "${CMSSW_BASE}/src/PhysicsTools/NanoAODTools/data/jme/Fall17_V3_MC_SF_AK4PFchs.txt",
    2018: "${CMSSW_BASE}/src/PhysicsTools/NanoAODTools/data/jme/Autumn18_V7_MC_SF_AK4PFchs.txt"
}

#analyzerChain.append(
#     MetFilter(
#        globalOptions=globalOptions,
#        outputName="MET_filter"
#     )
#)

def jetSelectionSequence(jetDict):
    sequence = []
    for systName,jetCollection in jetDict.items():
        sequence.extend([
            JetSelection(
                inputCollection=jetCollection,
                #leptonCollectionDRCleaning=lambda event: event.LooseMuons,
                #leptonCollectionP4Subtraction=lambda event:event.LooseMuons,
                globalFeatures = ['numberCpf', 'numberMuon', 'numberElectron'],
                outputName="selectedJets_"+systName,
            ),
        ])
#        
# GEN-level label for jet
#        if isMC:
#            sequence.append(
#                JetTruthFlags(
#                    inputCollection=lambda event, systName=systName: getattr(event, "selectedJets_"+systName),
#                    originVariables = ['displacement_xy'],
#                    outputName="selectedJets_"+systName,
#                    globalOptions=globalOptions
#                )
#            )
#    
#    systNames = jetDict.keys()
#    sequence.append(
#        EventSkim(selection=lambda event, systNames=systNames: 
#            any([getattr(event, "nselectedJets_"+systName) > 0 for systName in systNames]),
#            outputName="jet",
#        )
#    )
#            
    return sequence
    
    
def eventReconstructionSequence(jetMetDict):
    sequence = []
    for systName,(jetCollection,metObject) in jetMetDict.items():
        sequence.extend([
            EventObservables(
                lepton1Object=lambda event: event.leadingLeptons[0],
                lepton2Object=None if args.leptons==1 else (lambda event: event.subleadingLeptons[0]),
                jetCollection=jetCollection,
                metInput=metObject,
                globalOptions=globalOptions,
                outputName=systName
            ),
            HNLReconstruction(
                lepton1Object=lambda event: event.leadingLeptons[0],
                lepton2Object=None if args.leptons==1 else (lambda event: event.subleadingLeptons[0]),
                jetCollection=jetCollection,
                globalOptions=globalOptions,
                outputName=systName,
            ),
        ])
    sequence.extend([
        TrackAndSVSelection(
            svType='regular',
            outputName="hnlJet_track_weight",
            jetCollection = lambda event: event.hnlJets_nominal,
            lepton2Object = None if args.leptons==1 else (lambda event: event.subleadingLeptons[0]),
            globalOptions=globalOptions
        )
    ])
        
    return sequence
    
    
def taggerSequence(jetDict, modelFile, taggerName):
    sequence = []
    if args.notagger:
        return []
    sequence.append(
        TaggerEvaluationProfiled(
            modelPath=taggerModelPath[year],
            featureDictFile=featureDictFile,
            inputCollections=jetDict.values(),
            taggerName=taggerName,
            profiledLabelDict = {
                'LLP_Q': ['LLP_Q','LLP_QTAU_H','LLP_QTAU_3H'],
                'LLP_QE': [ 'LLP_QE'],
                'LLP_QMU': [ 'LLP_QMU'],
                'LLP_QTAU_3H': ['LLP_QTAU_3H']
            },
            globalOptions=globalOptions,
            evalValues = np.linspace(-1.9,1.9,5*4),
        )
    )
   
    for systName,jetCollection in jetDict.items():
        sequence.append(
            JetTaggerProfiledResult(
                inputCollection=jetCollection,
                taggerName=taggerName,
                outputName="hnlJet_"+systName,
                profiledLabels = ['LLP_Q','LLP_QE','LLP_QMU','LLP_QTAU_3H'],
                kinds = ['single','ratio'],
                globalOptions={"isData": False}
            )
        )
    return sequence

    

def bdtSequence(systematics):
    sequence = []
    if args.nobdt:
        return []
    if args.leptons==1:
        sequence.append(
            XGBEvaluation(
                systematics=systematics,
                modelPath=BDT1lmodelPathUncorr[year],
                inputFeatures="${CMSSW_BASE}/src/PhysicsTools/NanoAODTools/data/bdt/201117/uncorrelated/bdt_1l_inputs.py",
                outputName="bdt_score"
            )
        )
    else:
        sequence.append(
            XGBEvaluation(
                systematics=systematics,
                modelPath=BDT2lmodelPathExperimental[year],
                inputFeatures="${CMSSW_BASE}/src/PhysicsTools/NanoAODTools/data/bdt/201117/experimental/bdt_2l_inputs.py",
                outputName="bdt_score"
            )
        )
    return sequence

# returns event weight per /fb for QCD mu-enriched Shat-binned MC samples
def qcdShatWeight(fileName):
    xs_pb = {
        "Pt-15to20": 2799000.0,
        "Pt-20to30": 2526000.0,
        "Pt-30to50": 1362000.0,
        "Pt-50to80": 376600.0,
        "Pt-80to120": 88930.0,
        "Pt-120to170": 21230.0,
        "Pt-170to300": 7055.0,
        "Pt-300to470": 619.3,
        "Pt-470to600": 59.24,
        "Pt-600to800": 18.21,
        "Pt-800to1000": 3.275,
        "Pt-1000toInf": 1.078,
    }
    weight = 1.
    if "QCD_" in args.inputFiles[0] and "_MuEnrichedPt5" in args.inputFiles[0]:
        pt_bin = str(args.inputFiles[0].split("QCD_")[1].split("_MuEnrichedPt5")[0])
        xs = xs_pb.get(pt_bin,None)
        tree.GetEntry(0); nevents = tree.totalBeforeSkim
        weight = 1000. * xs / nevents if xs is not None else -1.
    return weight

if isMC:
#    analyzerChain.append(
#        JetMetUncertainties(
#            metInput= lambda event: Object(event, met_variable[year]),
#            rhoInput = lambda event: event.fixedGridRhoFastjetAll,
#            jetCollection = lambda event: Collection(event,"Jet"),
#            lowPtJetCollection = lambda event: Collection(event,"CorrT1METJet"),
#            genJetCollection = lambda event: Collection(event,"GenJet"),
#            muonCollection = lambda event: Collection(event,"Muon"),
#            electronCollection = lambda event: Collection(event,"Electron"),
#            jesUncertaintyFile=jesUncertaintyFile[year],
#            jerResolutionFileName=jerResolutionFile[year],
#            jerSFUncertaintyFileName=jerSFUncertaintyFile[year],
#            propagateJER = False,
#            jetKeys = ['jetId', 'nConstituents', 'rawFactor'],
#        )
#    )

#    analyzerChain.append(
#        PhiXYCorrection(
#            era=str(year),
#            metInputDict={
#                "met_nominal": "met_nominal", 
#                #"met_jerUp": "met_jerUp",
#                #"met_jerDown": "met_jerDown",
#                #"met_jesTotalUp": "met_jesTotalUp",
#                #"met_unclEnUp": "met_unclEnUp",
#                #"met_unclEnDown": "met_unclEnDown"
#            },
#            isMC=isMC,
#            )
#        )

    analyzerChain.extend(
        jetSelectionSequence({
            "nominal": lambda event: Collection(event,"Jet"),#jets_nominal,
#            #"jerUp": lambda event: event.jets_jerUp,
#            #"jerDown": lambda event: event.jets_jerDown,
#            #"jesTotalUp": lambda event: event.jets_jesTotalUp,
#            #"jesTotalDown": lambda event: event.jets_jesTotalDown,
        })
    )
        
        
#    analyzerChain.extend(
#        eventReconstructionSequence({
#            "nominal": (lambda event: event.selectedJets_nominal, lambda event: event.met_nominal),
##            "jerUp": (lambda event: event.selectedJets_jerUp, lambda event: event.met_jerUp),
##            "jerDown": (lambda event: event.selectedJets_jerDown, lambda event: event.met_jerDown),
##            "jesTotalUp": (lambda event: event.selectedJets_jesTotalUp, lambda event: event.met_jesTotalUp),
##            "jesTotalDown": (lambda event: event.selectedJets_jesTotalDown, lambda event: event.met_jesTotalDown),
##            "unclEnUp": (lambda event: event.selectedJets_nominal, lambda event: event.met_unclEnUp),
##            "unclEnDown": (lambda event: event.selectedJets_nominal, lambda event: event.met_unclEnDown),
#        })
#    )    

#    analyzerChain.append(
#        HEMFlag(
#            inputDict={
#            "nominal": lambda event: event.hnlJets_nominal,
#            "jerUp": lambda event: event.hnlJets_jerUp,
#            "jerDown": lambda event: event.hnlJets_jerDown,
#            "jesTotalUp": lambda event: event.hnlJets_jesTotalUp,
#            "jesTotalDown": lambda event: event.hnlJets_jesTotalDown
#        },
#            leadingLeptons=lambda event: event.leadingLeptons,
#            subleadingLeptons=lambda event: event.subleadingLeptons
#        )
#    )    
    
#    analyzerChain.append(
#        PileupWeight(
#            outputName="puweight",
#            processName=puProcessName,
#            globalOptions=globalOptions
#        )
#    )

    pass#@@

else:
    analyzerChain.append(
        PhiXYCorrection(
            era=str(year),
            metInputDict={"met_nominal": met_variable[year]}, 
            isMC=isMC,
            metObject=True
            )
        )

    analyzerChain.extend(
        jetSelectionSequence({
            "nominal": lambda event: Collection(event,"Jet")
        })
    )
    
    analyzerChain.extend(
        eventReconstructionSequence({
            "nominal": (lambda event: event.selectedJets_nominal, lambda event: Object(event, met_variable[year])),
        })
    )
    analyzerChain.append(
        HEMFlag(
            inputDict={
            "nominal": lambda event: event.hnlJets_nominal,
        },
            leadingLeptons=lambda event: event.leadingLeptons,
            subleadingLeptons=lambda event: event.subleadingLeptons
        )
    )    
    
    analyzerChain.extend(
        taggerSequence({
            "nominal": lambda event: event.hnlJets_nominal,
        },
        modelFile=taggerModelPath[year],
        taggerName='llpdnnx'
    ))
   
    analyzerChain.extend(
        bdtSequence([
            "nominal",
        ])
    )


storeVariables = [
    [lambda tree: tree.branch("nSV", "I"), 
     lambda tree,event: tree.fillBranch("nSV",event.nSV)],
    [lambda tree: tree.branch("SV_mass", "F", lenVar="nSV"), 
     lambda tree,event: tree.fillBranch("SV_mass",[event.SV_mass[i] for i in range(len(event.SV_mass))])],
]

weight = qcdShatWeight(args.inputFiles[0])
storeVariables.append([lambda tree: tree.branch("qcdShatWeight", "F"),
                       lambda tree, event: tree.fillBranch("qcdShatWeight",weight)])

analyzerChain.append(EventInfo(storeVariables=storeVariables))

p = PostProcessor(
    args.output[0],
    [args.inputFiles],
    modules=analyzerChain,
    maxEvents=-1,
    friend=True,
    #cut="((nMuon)>0)", # remove if doing cutflow
    cutFlow=args.cutflow
)

p.run()

