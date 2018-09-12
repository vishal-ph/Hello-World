library(oro.nifti)

args <- commandArgs(trailingOnly = TRUE)
inputFileName = (args[1])
TR = as.double(args[2])
targetTime = as.double(args[3])
sliceTimeFile = (args[4])
outputFileName = (args[5])

# TR = 720.0
# targetTime = 0.0
# inputFileName = "rest2.nii.gz"
# sliceTimeFile = "SliceTimeAcq.txt"
# outputFileName = "output"


outputFileName_text = paste(outputFileName,".txt", sep="")

#setwd("/Users/vishalsharma/R_Working_Directory/COL_Assignment2")

inputFileNIfTI = readNIfTI(inputFileName, reorient = FALSE)
dimension = dim(inputFileNIfTI)
inputFile = array(data = inputFileNIfTI, dim = dimension)
inputFile_corr = array( data = inputFileNIfTI, dim = dimension)


sliceAc = read.table(sliceTimeFile)     # imports like a table of mXn dimensions

error = 0.0

for(i in 1:length(sliceAc[,1])) {
  if((sliceAc[i,1]>TR) || (targetTime > TR)) error <- error + 1
}

if(error > 0.0) write("FAILURE", file=outputFileName_text) else write("SUCCESS", file=outputFileName_text)

if(error == 0) {
  for(z in 1:length(sliceAc[,1])) {
    t =sliceAc[z,1]#int(float(slice))
    for(k in 2:(dimension[4]-1)) {
      if(targetTime <= t ) {
        inputFile_corr[,,z,k] = inputFile[,,z,k] + ((targetTime - t)/TR) * (inputFile[,,z,k] - inputFile[,,z,k-1])
      } else {inputFile_corr[,,z,k] = inputFile[,,z,k] + ((targetTime - t)/TR) * (inputFile[,,z,k+1] - inputFile[,,z,k])}   
    }
  }  
  outputFileCorr = nifti(inputFile_corr, datatype = 16)
  writeNIfTI(outputFileCorr, filename = outputFileName)
}

#trial = readNIfTI("trial.nii.gz", reorient = FALSE)

