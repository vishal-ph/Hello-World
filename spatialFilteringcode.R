library(oro.nifti)
setwd("/Users/vishalsharma/R_workD/COL_Assignment4")

inputFileNIfTI = readNIfTI("rest.nii.gz", reorient = FALSE)
temp = readNIfTI("temporal_TR2.5_whole.nii.gz", reorient = FALSE)
spatial = readNIfTI("spatial_TR2.5_whole.nii.gz", reorient = FALSE)

#inputFileNIfTI = readNIfTI(inputFileName, reorient = FALSE)
dimension = dim(inputFileNIfTI)
inputFile = array(data = inputFileNIfTI, dim = dimension)
inputFile_corr = array( data = 0, dim = dimension)

Gaussain <- function(x,y,z,FWHM){
  sigma = FWHM/(2*sqrt(2*log(2)))
  G = 1/((2*pi)^(3/2)*sigma^3)*exp(-(x^2 + y^2 + z^2)/2*(sigma^2))
  return(G)
}

#PRECOMPUTE GAUSSIAN KERNEL
#RESTRICT THE KERNEL SIZE FOR ERROR OF 1E-3 +-3 OR +-4
#3D CONVOLUTION AS 3 1D CONVOLUTIONS--- OR--- 3D CONVOLUTION AS SPARSE MATRIX VECTOR PRODUCT--- OR--- 3D CONV. USING FFT.


FWHM = 3
sigma = FWHM/(2*sqrt(2*log(2)))

for(i in 1:dimension[1]){
  for(j in 1:dimension[2]){
    for(k in 1:dimension[3]){
      x = -1
      while(x <= 1 && i-x >= 1 && i-x <= dimension[1]) {
        y = -1
        while (y <= 1 && j-y >= 1 && j-y <= dimension[2]) {
          z = -1
          while (z <= 1 && k-z >= 1 && k-z <= dimension[3]) {
            G = 1/((2*pi)^(3/2)*sigma^3)*exp(-(x^2 + y^2 + z^2)/2*(sigma^2))
            inputFile_corr[i,j,k,] = inputFile_corr[i,j,k,] + inputFile[i-x,j-y,k-z,]*G 
            z = z + 1
          }
          y = y + 1
        }
        x = x + 1
      }
      #inputFile_corr[x,y,z,] = inputFile[x,y,z,]
    }
  }
}