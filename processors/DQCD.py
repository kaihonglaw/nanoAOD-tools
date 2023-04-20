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
from PhysicsTools.NanoAODTools.postprocessing.modules.common.countHistogramsModule import countHistogramsModule

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

print("isData:",args.isData)
print("inputs:",len(args.inputFiles))
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
        print("CRITICAL - file '"+inputFile+"' not found!")
        sys.exit(1)
    tree = rootFile.Get("Events")
    if not tree:
        print("CRITICAL - 'Events' tree not found in file '"+inputFile+"'!")
        sys.exit(1)
    print(" - ", inputFile, ", events=", tree.GetEntries())

puProcessName = args.overwrite_pu

print("year:", year)
print("isSignal:",isSignal)
print("No lepton iso: ","True" if args.noiso is True else "False (default)")
print("No trigger: ","True" if args.notrigger is True else "False (default)")
print("No BDT: ","True" if args.nobdt is True else "False (default)")
print("No tagger: ","True" if args.notagger is True else "False (default)")
print("channel: ","single lepton" if args.leptons==1 else "dilepton")
print("output directory:", args.output[0])

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
        muonMaxEta=1.5,
        muonMinPt=9.,
        muonMinSip3d=6.,
    ),
    MuonSelection(
        inputCollection=lambda event: event.LooseMuons,
        outputName="MuonsWithEtaAndPtReq",
        muonMaxEta=2.4,
        muonMinPt=5,
    ),
    EventSkim(selection=lambda event: event.nLooseMuons > 0, outputName="Muons"),
    EventSkim(selection=lambda event: event.nMuonsWithEtaAndPtReq > 0, outputName="MuonsWithEtaPtReq"),
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

analyzerChain = [countHistogramsModule()]

analyzerChain.extend(leptonSelection)
'''
analyzerChain.extend([
    EventSkim(selection=lambda event: event.nMuon > 0, outputName="nMuons"),
    #EventSkim(selection=lambda event: np.sum(np.logical_and(np.logical_and(event.Muon_pt > 5.0, event.Muon_eta < 2.4), event.Muon_eta > -2.4)) > 0, outputName="Muons"),
    EventSkim(selection=lambda event: np.sum(event.Muon_eta > -2.4 & event.Muon_eta < 2.4) > 0, outputName="Muons")
    #EventSkim(selection=lambda event: np.sum(event.Muon_pt > 5.0 & event.Muon_eta < 2.4 & event.Muon_eta > -2.4) > 0, outputName="Muons")
])
'''

if args.notrigger is False:
    trigger_matched = lambda event: any([muon.isTriggering>0 and muon.fired_HLT_Mu9_IP6>0 for muon in event.TriggeringMuons])
    #trigger_matched2 = lambda event: sum([muon.isTriggerMatched>0 for muon in event.TriggeringMuons]) == 2
    analyzerChain.extend([
        EventSkim(selection=lambda event: (event.DisplacedMuonTrigger_flag) > 0,  outputName="l1_trigger"),
        EventSkim(selection=trigger_matched, outputName="l1_triggermatch"),
    ])

leptonSelection2=[
MuonSelection(
        inputCollection=lambda event: event.TriggeringMuons,
        outputName="TriggeringMuons_triggermatched",
        triggermatching = True,
    ),
]

leptonSelection3=[
MuonSelection(
        inputCollection=lambda event: event.LooseMuons,
        outputName="MuonsWithTighterEtaAndPtReq",
        muonMaxEta=1.5,
        muonMinPt=10,
    ),
]

analyzerChain.extend(leptonSelection2)
analyzerChain.extend(leptonSelection3)

'''
if args.notrigger is False:
    analyzerChain.append(
	EventSkim(selection=lambda event: (event.DisplacedMuonTrigger_flag) > 0,  outputName="l1_trigger")
    )    
'''
'''    
analyzerChain.append(
    EventSkim(selection=lambda event: event.nLooseMuons>=args.nLeptons, outputName="MinMuons")
)
'''


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
                jetMinPt=15.,
                jetMaxEta=2.4,
                globalFeatures = ['numberCpf', 'numberMuon', 'numberElectron'],
                outputName="selectedJets_"+systName,
                #outputName="selectedJets",
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
    '''
    if "QCD_" in args.inputFiles[0] and "_MuEnrichedPt5" in args.inputFiles[0]:
        pt_bin = str(args.inputFiles[0].split("QCD_")[1].split("_MuEnrichedPt5")[0])
        xs = xs_pb.get(pt_bin,None)
        tree.GetEntry(0); nevents = tree.totalBeforeSkim
        weight = 1000. * xs / nevents if xs is not None else -1.
    '''
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
        
    analyzerChain.append(
        EventSkim(selection=lambda event: event.nselectedJets_nominal > 0, outputName="JetswithEtaPtReq")
    ) 
     
    analyzerChain.append(
        EventSkim(selection=lambda event: event.xgb0__m_2p0_ctau_10p0_xiO_1p0_xiL_1p0 >= 0.0, outputName="BDTscore")
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
    [lambda tree: tree.branch("SV_dxy", "F", lenVar="nSV"), 
     lambda tree,event: tree.fillBranch("SV_dxy",[event.SV_dxy[i] for i in range(len(event.SV_dxy))])],
    [lambda tree: tree.branch("SV_dxySig", "F", lenVar="nSV"), 
     lambda tree,event: tree.fillBranch("SV_dxySig",[event.SV_dxySig[i] for i in range(len(event.SV_dxySig))])],
    [lambda tree: tree.branch("nsv", "I"), 
     lambda tree,event: tree.fillBranch("nsv",event.nsv)],
    [lambda tree: tree.branch("sv_mass", "F", lenVar="nsv"), 
     lambda tree,event: tree.fillBranch("sv_mass",[event.sv_mass[i] for i in range(len(event.sv_mass))])],
    [lambda tree: tree.branch("sv_dxy", "F", lenVar="nsv"), 
     lambda tree,event: tree.fillBranch("sv_dxy",[event.sv_dxy[i] for i in range(len(event.sv_dxy))])],
    [lambda tree: tree.branch("sv_dxysig", "F", lenVar="nsv"), 
     lambda tree,event: tree.fillBranch("sv_dxysig",[event.sv_dxysig[i] for i in range(len(event.sv_dxysig))])],
    [lambda tree: tree.branch("sv_ntracks", "F", lenVar="nsv"),
     lambda tree,event: tree.fillBranch("sv_ntracks",[event.sv_ntracks[i] for i in range(len(event.sv_ntracks))])],   
    [lambda tree: tree.branch("nsvAdapted", "I"), 
     lambda tree,event: tree.fillBranch("nsvAdapted",event.nsvAdapted)],
    [lambda tree: tree.branch("svAdapted_mass", "F", lenVar="nsvAdapted"), 
     lambda tree,event: tree.fillBranch("svAdapted_mass",[event.svAdapted_mass[i] for i in range(len(event.svAdapted_mass))])],
    [lambda tree: tree.branch("svAdapted_dxy", "F", lenVar="nsvAdapted"), 
     lambda tree,event: tree.fillBranch("svAdapted_dxy",[event.svAdapted_dxy[i] for i in range(len(event.svAdapted_dxy))])],
    [lambda tree: tree.branch("svAdapted_dxysig", "F", lenVar="nsvAdapted"), 
     lambda tree,event: tree.fillBranch("svAdapted_dxysig",[event.svAdapted_dxysig[i] for i in range(len(event.svAdapted_dxysig))])],
    [lambda tree: tree.branch("svAdapted_ntracks", "F", lenVar="nsvAdapted"),
     lambda tree,event: tree.fillBranch("svAdapted_ntracks",[event.svAdapted_ntracks[i] for i in range(len(event.svAdapted_ntracks))])],
    [lambda tree: tree.branch("nMuon", "I"),
     lambda tree,event: tree.fillBranch("nMuon",event.nMuon)],
    [lambda tree: tree.branch("Muon_sip3d", "F", lenVar="nMuon"),
     lambda tree,event: tree.fillBranch("Muon_sip3d",[event.Muon_sip3d[i] for i in range(len(event.Muon_sip3d))])],
    [lambda tree: tree.branch("Muon_ip3d", "F", lenVar="nMuon"),
     lambda tree,event: tree.fillBranch("Muon_ip3d",[event.Muon_ip3d[i] for i in range(len(event.Muon_ip3d))])],
    [lambda tree: tree.branch("Muon_dxy", "F", lenVar="nMuon"),
     lambda tree,event: tree.fillBranch("Muon_dxy",[event.Muon_dxy[i] for i in range(len(event.Muon_dxy))])],
    [lambda tree: tree.branch("Muon_dz", "F", lenVar="nMuon"),
     lambda tree,event: tree.fillBranch("Muon_dz",[event.Muon_dz[i] for i in range(len(event.Muon_dz))])],
    [lambda tree: tree.branch("Muon_pt", "F", lenVar="nMuon"),
     lambda tree,event: tree.fillBranch("Muon_pt",[event.Muon_pt[i] for i in range(len(event.Muon_pt))])],
    [lambda tree: tree.branch("Muon_dzErr", "F", lenVar="nMuon"),
     lambda tree,event: tree.fillBranch("Muon_dzErr",[event.Muon_dzErr[i] for i in range(len(event.Muon_dzErr))])],
    [lambda tree: tree.branch("Muon_dxyErr", "F", lenVar="nMuon"),
     lambda tree,event: tree.fillBranch("Muon_dxyErr",[event.Muon_dxyErr[i] for i in range(len(event.Muon_dxyErr))])],
    [lambda tree: tree.branch("Muon_eta", "F", lenVar="nMuon"),
     lambda tree,event: tree.fillBranch("Muon_eta",[event.Muon_eta[i] for i in range(len(event.Muon_eta))])],
    [lambda tree: tree.branch("Muon_phi", "F", lenVar="nMuon"),
     lambda tree,event: tree.fillBranch("Muon_phi",[event.Muon_phi[i] for i in range(len(event.Muon_phi))])],
    [lambda tree: tree.branch("Muon_charge", "I", lenVar="nMuon"),
     lambda tree,event: tree.fillBranch("Muon_charge",[event.Muon_charge[i] for i in range(len(event.Muon_charge))])],
    [lambda tree: tree.branch("nJet", "I"),
     lambda tree,event: tree.fillBranch("nJet",event.nJet)],
    [lambda tree: tree.branch("Jet_muEF", "F", lenVar="nJet"),
     lambda tree,event: tree.fillBranch("Jet_muEF",[event.Jet_muEF[i] for i in range(len(event.Jet_muEF))])],
    [lambda tree: tree.branch("Jet_muonSubtrFactor", "F", lenVar="nJet"),
     lambda tree,event: tree.fillBranch("Jet_muonSubtrFactor",[event.Jet_muonSubtrFactor[i] for i in range(len(event.Jet_muonSubtrFactor))])],
    [lambda tree: tree.branch("Jet_pt", "F", lenVar="nJet"),
     lambda tree,event: tree.fillBranch("Jet_pt",[event.Jet_pt[i] for i in range(len(event.Jet_pt))])],
    [lambda tree: tree.branch("Leading_Jet_pt", "F"),
     lambda tree,event: tree.fillBranch("Leading_Jet_pt",event.Jet_pt[0])],
    [lambda tree: tree.branch("nmuonSV", "I"),
     lambda tree,event: tree.fillBranch("nmuonSV",event.nmuonSV)],
    [lambda tree: tree.branch("muonSV_dlen", "F", lenVar="nmuonSV"),
     lambda tree,event: tree.fillBranch("muonSV_dlen",[event.muonSV_dlen[i] for i in range(len(event.muonSV_dlen))])],
    [lambda tree: tree.branch("muonSV_dlenSig", "F", lenVar="nmuonSV"),
     lambda tree,event: tree.fillBranch("muonSV_dlenSig",[event.muonSV_dlenSig[i] for i in range(len(event.muonSV_dlenSig))])],
    [lambda tree: tree.branch("muonSV_dxy", "F", lenVar="nmuonSV"),
     lambda tree,event: tree.fillBranch("muonSV_dxy",[event.muonSV_dxy[i] for i in range(len(event.muonSV_dxy))])],
    [lambda tree: tree.branch("muonSV_dxySig", "F", lenVar="nmuonSV"),
     lambda tree,event: tree.fillBranch("muonSV_dxySig",[event.muonSV_dxySig[i] for i in range(len(event.muonSV_dxySig))])],
    [lambda tree: tree.branch("muonSV_x", "F", lenVar="nmuonSV"),
     lambda tree,event: tree.fillBranch("muonSV_x",[event.muonSV_x[i] for i in range(len(event.muonSV_x))])],
    [lambda tree: tree.branch("muonSV_y", "F", lenVar="nmuonSV"),
     lambda tree,event: tree.fillBranch("muonSV_y",[event.muonSV_y[i] for i in range(len(event.muonSV_y))])],
    [lambda tree: tree.branch("muonSV_z", "F", lenVar="nmuonSV"),
     lambda tree,event: tree.fillBranch("muonSV_z",[event.muonSV_z[i] for i in range(len(event.muonSV_z))])],
    [lambda tree: tree.branch("muonSV_ndof", "F", lenVar="nmuonSV"),
     lambda tree,event: tree.fillBranch("muonSV_ndof",[event.muonSV_ndof[i] for i in range(len(event.muonSV_ndof))])], 
    [lambda tree: tree.branch("muonSV_chi2", "F", lenVar="nmuonSV"),
     lambda tree,event: tree.fillBranch("muonSV_chi2",[event.muonSV_chi2[i] for i in range(len(event.muonSV_chi2))])],
    [lambda tree: tree.branch("muonSV_pAngle", "F", lenVar="nmuonSV"),
     lambda tree,event: tree.fillBranch("muonSV_pAngle",[event.muonSV_pAngle[i] for i in range(len(event.muonSV_pAngle))])],
    [lambda tree: tree.branch("muonSV_origMass", "F", lenVar="nmuonSV"),
     lambda tree,event: tree.fillBranch("muonSV_origMass",[event.muonSV_origMass[i] for i in range(len(event.muonSV_origMass))])],
    [lambda tree: tree.branch("muonSV_mass", "F", lenVar="nmuonSV"),
     lambda tree,event: tree.fillBranch("muonSV_mass",[event.muonSV_mass[i] for i in range(len(event.muonSV_mass))])],
    [lambda tree: tree.branch("muonSV_mu1pt", "F", lenVar="nmuonSV"),
     lambda tree,event: tree.fillBranch("muonSV_mu1pt",[event.muonSV_mu1pt[i] for i in range(len(event.muonSV_mu1pt))])],
    [lambda tree: tree.branch("muonSV_mu2pt", "F", lenVar="nmuonSV"),
     lambda tree,event: tree.fillBranch("muonSV_mu2pt",[event.muonSV_mu2pt[i] for i in range(len(event.muonSV_mu2pt))])],
    [lambda tree: tree.branch("muonSV_mu1phi", "F", lenVar="nmuonSV"),
     lambda tree,event: tree.fillBranch("muonSV_mu1phi",[event.muonSV_mu1phi[i] for i in range(len(event.muonSV_mu1phi))])],
    [lambda tree: tree.branch("muonSV_mu2phi", "F", lenVar="nmuonSV"),
     lambda tree,event: tree.fillBranch("muonSV_mu2phi",[event.muonSV_mu2phi[i] for i in range(len(event.muonSV_mu2phi))])],
    [lambda tree: tree.branch("muonSV_mu1eta", "F", lenVar="nmuonSV"),
     lambda tree,event: tree.fillBranch("muonSV_mu1eta",[event.muonSV_mu1eta[i] for i in range(len(event.muonSV_mu1eta))])],
    [lambda tree: tree.branch("muonSV_mu2eta", "F", lenVar="nmuonSV"),
     lambda tree,event: tree.fillBranch("muonSV_mu2eta",[event.muonSV_mu2eta[i] for i in range(len(event.muonSV_mu2eta))])],
    [lambda tree: tree.branch("muonSV_mu1index", "F", lenVar="nmuonSV"),
     lambda tree,event: tree.fillBranch("muonSV_mu1index",[event.muonSV_mu1index[i] for i in range(len(event.muonSV_mu1index))])],
    [lambda tree: tree.branch("muonSV_mu2index", "F", lenVar="nmuonSV"),
     lambda tree,event: tree.fillBranch("muonSV_mu2index",[event.muonSV_mu2index[i] for i in range(len(event.muonSV_mu2index))])],
    [lambda tree: tree.branch("xgb0__m_2p0_ctau_10p0_xiO_1p0_xiL_1p0", "F"),
     lambda tree,event: tree.fillBranch("xgb0__m_2p0_ctau_10p0_xiO_1p0_xiL_1p0",event.xgb0__m_2p0_ctau_10p0_xiO_1p0_xiL_1p0)],
    [lambda tree: tree.branch("xgb0_no_mass__m_2p0_ctau_10p0_xiO_1p0_xiL_1p0", "F"),
     lambda tree,event: tree.fillBranch("xgb0_no_mass__m_2p0_ctau_10p0_xiO_1p0_xiL_1p0",event.xgb0_no_mass__m_2p0_ctau_10p0_xiO_1p0_xiL_1p0)]   
]
'''
storeVariables.append([lambda tree: tree.branch("Subleading_Jet_pt", "F"),
                       lambda tree,event: tree.fillBranch("Subleading_Jet_pt",[event.Jet_pt[1]]) if (event.nJet > 1)])
'''
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

