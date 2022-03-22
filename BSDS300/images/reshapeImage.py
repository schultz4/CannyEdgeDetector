import numpy as np
from PIL import Image
import os 

foldername  = '/Users/catmcqueen/Documents/SPR22/HPC/proj/CannyEdgeDetector/BSDS300/images/train'
imagefolder = os.listdir(foldername)  

for im in imagefolder:
    image 	= Image.open(im)
    newimg 	= image.resize((300, 300))
    filenm 	= im[:-4] + '_resized' + im[-4:]
    newimg.save(filenm)





