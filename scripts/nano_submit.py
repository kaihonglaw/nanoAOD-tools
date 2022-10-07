import argparse
import os,stat
from glob import glob
parser = argparse.ArgumentParser()
parser.add_argument("-i", "--inputDir", help="Input directory")
parser.add_argument("-b", "--BDTDir", help="BDT directory")
parser.add_argument("-o", "--outputDir", help="Output directory")
args = parser.parse_args()

def submit(inputDir, BDTDir, outputDir):
    print (inputDir, BDTDir, outputDir)
    #inputFiles = glob(inputDir+"/*.root")
    inputFilesBDT = glob(BDTDir+"/*.root") 
    inputFiles = [os.path.join(inputDir,x.split("/")[-1].replace("-icenet","")) for x in inputFilesBDT]
    outputFiles = [os.path.join(outputDir,x.split("/")[-1]) for x in inputFiles]
    print("inputFiles =",inputFilesBDT)
    submitDir = "tasks/{}".format("task"+inputDir.replace("/","_"))
    if not os.path.exists(submitDir):
        os.makedirs(submitDir)
    if not os.path.exists(outputDir):
        os.makedirs(outputDir)
    submitBash = submitDir+"/submitScript.sh"
    with open(submitBash,"w") as f:
        iF = 0
        f.write('''
source /cvmfs/cms.cern.ch/cmsset_default.sh
cd /vols/cms/khl216/CMSSW_10_2_18
eval $(scramv1 runtime -sh)
cd -
''')
        for ifile, ifileBDT, ofile in zip(inputFiles, inputFilesBDT, outputFiles):
            iF += 1
            f.write("fileArray[{}]='python /vols/cms/khl216/CMSSW_10_2_18/src/PhysicsTools/NanoAODTools/processors/DQCD.py --input {} --input {} {}'\n".format(iF,ifile, ifileBDT, outputDir))
        f.write("eval ${fileArray[$SGE_TASK_ID]}\n")
    st = os.stat(submitBash)
    os.chmod(submitBash, st.st_mode | stat.S_IEXEC)
    import subprocess
    bashCommand = "cd {} && qsub -q hep.q -l h_rt=3600 -t 1-{}:1 -cwd {} && cd -".format(submitDir,iF,submitBash.split("/")[-1])
    os.system(bashCommand)
    # print bashCommand.split()
    # process = subprocess.run(bashCommand.split(), stdout=subprocess.PIPE)
    # output, error = process.stdout,process.stderr
'''
for ifile in glob("/vols/cms/mc3909/bparkProductionV1_bkg/*"):
    submit(ifile,ifile.replace("mc3909/","mc3909/SkimsV1/"))
'''
if __name__=="__main__":
	submit(**vars(args)) 

