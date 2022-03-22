import numpy as np
from PIL import Image
import os 

foldername  = '/Users/catmcqueen/Documents/SPR22/HPC/proj/CannyEdgeDetector/BSDS300/images/train/resized'
imagefolder = os.listdir(foldername)  

for im in imagefolder:
    try:
    	filename    = foldername + '/' + im
    	image 	= Image.open(filename)
    	filenm 	= foldername + '/png/' +  im[:-4] + '.png'
    	image.save(filenm)

    except:
        continue
