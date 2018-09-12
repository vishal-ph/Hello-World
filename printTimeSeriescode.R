args <- commandArgs(trailingOnly = TRUE)
inputFile = (args[1])
xCoordinate = as.integer(args[2])
yCoordinate = as.integer(args[3])
zCoordinate = as.integer(args[4])
outputFile = (args[5])

library(oro.nifti)

if(grepl("nii", inputFile)){
  InputFile = readNIfTI(inputFile)
} else{ 
  InputFile = readANALYZE(inputFile)
}

Tdim = dim_(InputFile)[5]

voxelTimeSeries = as.integer(InputFile[xCoordinate, yCoordinate, zCoordinate,])

write(voxelTimeSeries, file=outputFile , ncolumns = Tdim, sep = " ")


