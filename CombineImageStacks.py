# -*- coding: utf-8 -*-
"""
Created on Tue May 29 21:05:05 2018

@author: Matthew Peek
Last Modified: 16 December 2018
All Fields Image Stack
"""
import numpy as np
from scipy import stats
from astropy.io import ascii
import astropy.io.fits as fits
import scipy.ndimage as ndimage
from matplotlib import pyplot as plt
from photutils import CircularAnnulus
from photutils import CircularAperture, aperture_photometry
    
"""
StackAll function, takes numpy array as argument, loops through array argument
getting fits data and combining all image data into same array.

Stack image data standard, write results to new fits files.
"""
def stackAll(fileListAll):
    imageData = [fits.getdata(file) for file in fileListAll]
    #imageData = [file for file in fileList]
    
    print ("Total image data:", imageData,'\n')
    imageStack = np.sum(imageData, axis=0)
    
    fits.writeto('Stacked_Image_All.fits', imageStack, overwrite=True)
    
    print ("Image Sum:", imageStack,'\n')
    
    plt.clf()
    plt.imshow(imageStack)
    plt.savefig('Stacked_Image_All', dpi=100)
    plt.subplots_adjust(right=2.0)
    plt.subplots_adjust(top=1.0)
    plt.colorbar()
    plt.show()
    
    #Gaussian Blur
    plt.clf()
    betterImage = ndimage.gaussian_filter(imageStack, sigma=(2,2), order=0)
    plt.imshow(betterImage)
    plt.savefig('Stacked_Image_All_Blur', dpi=100)
    plt.subplots_adjust(right=2.0)
    plt.subplots_adjust(top=1.0)
    plt.colorbar()
    plt.show()
    
    print ("StackAll Function Complete!", '\n')
#End StackAll function
    

"""
StackMeanAll function, takes numpy array as argument, loops through array argument
getting fits data and combining all image data into same array.

Stack image data mean, write results to new fits files.

""" 
def stackMeanAll(fileListMeanAll):
    imageData = [fits.getdata(file) for file in fileListMeanAll]
    print ("Total image data:", imageData,'\n')
    meanImage = np.mean(imageData, axis=0)
    
    fits.writeto('Stacked_Image_Mean_All.fits', meanImage, overwrite=True)
    
    print ("Image Mean:", meanImage,'\n')
    
    plt.clf()
    plt.imshow(meanImage)
    plt.savefig('Stacked_Image_Mean_All', dpi=100)
    plt.subplots_adjust(right=2.0)
    plt.subplots_adjust(top=1.0)
    plt.colorbar()
    plt.show()
    
    #Gaussian Blur
    plt.clf()
    betterImage = ndimage.gaussian_filter(meanImage, sigma=(2,2), order=0)
    plt.imshow(betterImage)
    plt.savefig('Stacked_Image_Mean_All_Blur', dpi=100)
    plt.subplots_adjust(right=2.0)
    plt.subplots_adjust(top=1.0)
    plt.colorbar()
    plt.show()
    
    print ("stackMeanAll Function Complete!", '\n')
#End stackMeanAll function
    

"""
StackMedianAll function, takes numpy array as argument, loops through array argument
getting fits data and combining all image data into same array.

Stack image data median, write results to new fits files.
"""    
def stackMedianAll(fileListMedianAll):
    imageData = [fits.getdata(file) for file in fileListMedianAll]
    print ("Total image data:", imageData,'\n')
    medianImage = np.median(imageData, axis=0)
    
    fits.writeto('Stacked_Image_Median_All.fits', medianImage, overwrite=True)
    
    print ("Image Median:", medianImage,'\n')
    
    plt.clf()
    plt.imshow(medianImage)
    plt.savefig('Stacked_Image_Median_All', dpi=100)
    plt.subplots_adjust(right=2.0)
    plt.subplots_adjust(top=1.0)
    plt.colorbar()
    plt.show()
    
    #Gaussian Blur
    plt.clf()
    betterImage = ndimage.gaussian_filter(medianImage, sigma=(2,2), order=0)
    plt.imshow(betterImage)
    plt.savefig('Stacked_Image_Median_All_Blur', dpi=100)
    plt.subplots_adjust(right=2.0)
    plt.subplots_adjust(top=1.0)
    plt.colorbar()
    plt.show()
    
    print ("stackMedianAll Function Complete!")
#End stackMedianAll function


"""
StackMeanAbsorb function, takes numpy array as argument, loops through array argument
getting fits data and combining all image data into same array.

Stack image data median, write results to new fits files.
"""    
def stackMeanAbsorb(fileListMeanAbsorb):
    imageData = [fits.getdata(file) for file in fileListMeanAbsorb]
    print ("Total image data:", imageData,'\n')
    meanAbsorbImage = np.mean(imageData, axis=0)
    
    fits.writeto('Stacked_Image_Mean_Absorb.fits', meanAbsorbImage, overwrite=True)
    
    print ("Image Mean Absorb:", meanAbsorbImage,'\n')
    
    plt.clf()
    plt.imshow(meanAbsorbImage)
    plt.savefig('Stacked_Image_Mean_Absorb', dpi=100)
    plt.subplots_adjust(right=2.0)
    plt.subplots_adjust(top=1.0)
    plt.colorbar()
    plt.show()
    
    #Gaussian Blur
    plt.clf()
    betterImage = ndimage.gaussian_filter(meanAbsorbImage, sigma=(2,2), order=0)
    plt.imshow(betterImage)
    plt.savefig('Stacked_Image_Mean_Absorb_Blur', dpi=100)
    plt.subplots_adjust(right=2.0)
    plt.subplots_adjust(top=1.0)
    plt.colorbar()
    plt.show()
    
    print ("stackMeanAbsorb Function Complete!")
    return meanAbsorbImage
#End stackMeanAbsorb function
    
    
"""
StackMeanNonAbsorb function, takes numpy array as argument, loops through array argument
getting fits data and combining all image data into same array.

Stack image data median, write results to new fits files.
"""    
def stackMeanNonAbsorb(fileListMeanNonAbsorb):
    imageData = [fits.getdata(file) for file in fileListMeanNonAbsorb]
    print ("Total image data:", imageData,'\n')
    meanNonAbsorbImage = np.mean(imageData, axis=0)
    
    fits.writeto('Stacked_Image_Mean_NonAbsorb.fits', meanNonAbsorbImage, overwrite=True)
    
    print ("Image Mean Non-Absorb:", meanNonAbsorbImage,'\n')
    
    plt.clf()
    plt.imshow(meanNonAbsorbImage)
    plt.savefig('Stacked_Image_Mean_NonAbsorb', dpi=100)
    plt.subplots_adjust(right=2.0)
    plt.subplots_adjust(top=1.0)
    plt.colorbar()
    plt.show()
    
    #Gaussian Blur
    plt.clf()
    betterImage = ndimage.gaussian_filter(meanNonAbsorbImage, sigma=(2,2), order=0)
    plt.imshow(betterImage)
    plt.savefig('Stacked_Image_Mean_NonAbsorb_Blur', dpi=100)
    plt.subplots_adjust(right=2.0)
    plt.subplots_adjust(top=1.0)
    plt.colorbar()
    plt.show()
    
    print ("stackMeanNonAbsorb Function Complete!")
    return meanNonAbsorbImage
#End stackMeanNonAbsorb function

    
"""
StackMedianAbsorb function, takes numpy array as argument, loops through array argument
getting fits data and combining all image data into same array.

Stack image data median, write results to new fits files.
"""    
def stackMedianAbsorb(fileListMedianAbsorb):
    imageData = [fits.getdata(file) for file in fileListMedianAbsorb]
    print ("Total image data:", imageData,'\n')
    medianAbsorbImage = np.median(imageData, axis=0)
    
    fits.writeto('Stacked_Image_Median_Absorb.fits', medianAbsorbImage, overwrite=True)
    
    print ("Image Median Absorb:", medianAbsorbImage,'\n')
    
    plt.clf()
    plt.imshow(medianAbsorbImage)
    plt.savefig('Stacked_Image_Median_Absorb', dpi=100)
    plt.subplots_adjust(right=2.0)
    plt.subplots_adjust(top=1.0)
    plt.colorbar()
    plt.show()
    
    #Gaussian Blur
    plt.clf()
    betterImage = ndimage.gaussian_filter(medianAbsorbImage, sigma=(2,2), order=0)
    plt.imshow(betterImage)
    plt.savefig('Stacked_Image_Median_Absorb_Blur', dpi=100)
    plt.subplots_adjust(right=2.0)
    plt.subplots_adjust(top=1.0)
    plt.colorbar()
    plt.show()
    
    print ("stackMedianAbsorb Function Complete!")
#End stackMedianAbsorb function


"""
StackMedianNonAbsorb function, takes numpy array as argument, loops through array argument
getting fits data and combining all image data into same array.

Stack image data median, write results to new fits files.
"""    
def stackMedianNonAbsorb(fileListMedianNonAbsorb):
    imageData = [fits.getdata(file) for file in fileListMedianNonAbsorb]
    print ("Total image data:", imageData,'\n')
    medianNonAbsorbImage = np.median(imageData, axis=0)
    
    fits.writeto('Stacked_Image_Median_NonAbsorb.fits', medianNonAbsorbImage, overwrite=True)
    
    print ("Image Median Non-Absorb:", medianNonAbsorbImage,'\n')
    
    plt.clf()
    plt.imshow(medianNonAbsorbImage)
    plt.savefig('Stacked_Image_Median_NonAbsorb', dpi=100)
    plt.subplots_adjust(right=2.0)
    plt.subplots_adjust(top=1.0)
    plt.colorbar()
    plt.show()
    
    #Gaussian Blur
    plt.clf()
    betterImage = ndimage.gaussian_filter(medianNonAbsorbImage, sigma=(2,2), order=0)
    plt.imshow(betterImage)
    plt.savefig('Stacked_Image_Median_NonAbsorb_Blur', dpi=100)
    plt.subplots_adjust(right=2.0)
    plt.subplots_adjust(top=1.0)
    plt.colorbar()
    plt.show()
    
    print ("stackMedianNonAbsorb Function Complete!")
#End stackMedianNonAbsorb function


"""
StackStandardAbsorb function, takes numpy array as argument, loops through array argument
getting fits data and combining all image data into same array.

Stack image data median, write results to new fits files.
"""    
def stackStandardAbsorb(fileListStandardAbsorb):
    imageData = [fits.getdata(file) for file in fileListStandardAbsorb]
    print ("Total image data:", imageData,'\n')
    standardAbsorbImage = np.sum(imageData, axis=0)
    
    fits.writeto('Stacked_Image_Standard_Absorb.fits', standardAbsorbImage, overwrite=True)
    
    print ("Image Standard Absorb:", standardAbsorbImage,'\n')
    
    plt.clf()
    plt.imshow(standardAbsorbImage)
    plt.savefig('Stacked_Image_Standard_Absorb', dpi=100)
    plt.subplots_adjust(right=2.0)
    plt.subplots_adjust(top=1.0)
    plt.colorbar()
    plt.show()
    
    #Gaussian Blur
    plt.clf()
    betterImage = ndimage.gaussian_filter(standardAbsorbImage, sigma=(2,2), order=0)
    plt.imshow(betterImage)
    plt.savefig('Stacked_Image_Standard_Absorb_Blur', dpi=100)
    plt.subplots_adjust(right=2.0)
    plt.subplots_adjust(top=1.0)
    plt.colorbar()
    plt.show()
    
    print ("stackStandardAbsorb Function Complete!")
#End stackStandardAbsorb function


"""
StackStandardNonAbsorb function, takes numpy array as argument, loops through array argument
getting fits data and combining all image data into same array.

Stack image data median, write results to new fits files.
"""    
def stackStandardNonAbsorb(fileListStandardNonAbsorb):
    imageData = [fits.getdata(file) for file in fileListStandardNonAbsorb]
    print ("Total image data:", imageData,'\n')
    standardNonAbsorbImage = np.sum(imageData, axis=0)
    
    fits.writeto('Stacked_Image_Standard_NonAbsorb.fits', standardNonAbsorbImage, overwrite=True)
    
    print ("Image Standard Non-Absorb:", standardNonAbsorbImage,'\n')
    
    plt.clf()
    plt.imshow(standardNonAbsorbImage)
    plt.savefig('Stacked_Image_Standard_NonAbsorb', dpi=100)
    plt.subplots_adjust(right=2.0)
    plt.subplots_adjust(top=1.0)
    plt.colorbar()
    plt.show()
    
    #Gaussian Blur
    plt.clf()
    betterImage = ndimage.gaussian_filter(standardNonAbsorbImage, sigma=(2,2), order=0)
    plt.imshow(betterImage)
    plt.savefig('Stacked_Image_Standard_NonAbsorb_Blur', dpi=100)
    plt.subplots_adjust(right=2.0)
    plt.subplots_adjust(top=1.0)
    plt.colorbar()
    plt.show()
    
    print ("stackStandardNonAbsorb Function Complete!")
#End stackStandardNonAbsorb function
    
    
"""
reduceMeanImageStack function takes mean stacked absorber and mean stacked
non-absorber images as parameters. Subtracts noise out by finding difference
between the two. Writes reduced image to new fits file and makes .png image
with Gaussian blur applied.
"""
def reduceMeanImageStack(imageAbsorb, imageNonAbsorb):
    absorber = fits.getdata(imageAbsorb)
    nonAbsorber = fits.getdata(imageNonAbsorb)
    reduced = absorber - nonAbsorber
    fits.writeto('Reduced_Mean.fits', reduced, overwrite=True)
    
    plt.clf()
    plt.imshow(reduced)
    plt.savefig('Reduced_Mean', dpi=100)
    plt.subplots_adjust(right=2.0)
    plt.subplots_adjust(top=1.0)
    plt.colorbar()
    plt.show()

    #Gaussian Blur
    plt.clf()
    betterImage = ndimage.gaussian_filter(reduced, sigma=(2,2), order=0)
    plt.imshow(betterImage)
    plt.savefig('Reduced_Mean_Blur', dpi=100)
    plt.subplots_adjust(right=2.0)
    plt.subplots_adjust(top=1.0)
    plt.colorbar()
    plt.show()
    
    print ("Substract Mean Images Complete!")
#End reduceMeanImageStack function  
    

"""
reduceMedianImageStack function takes mean stacked absorber and mean stacked
non-absorber images as parameters. Subtracts noise out by finding difference
between the two. Writes reduced image to new fits file and makes .png image
with Gaussian blur applied.
"""
def reduceMedianImageStack(imageAbsorb, imageNonAbsorb):
    absorber = fits.getdata(imageAbsorb)
    nonAbsorber = fits.getdata(imageNonAbsorb)
    reduced = absorber - nonAbsorber
    fits.writeto('Reduced_Median.fits', reduced, overwrite=True)
    
    plt.clf()
    plt.imshow(reduced)
    plt.savefig('Reduced_Median', dpi=100)
    plt.subplots_adjust(right=2.0)
    plt.subplots_adjust(top=1.0)
    plt.colorbar()
    plt.show()

    #Gaussian Blur
    plt.clf()
    betterImage = ndimage.gaussian_filter(reduced, sigma=(2,2), order=0)
    plt.imshow(betterImage)
    plt.savefig('Reduced_Median_Blur', dpi=100)
    plt.subplots_adjust(right=2.0)
    plt.subplots_adjust(top=1.0)
    plt.colorbar()
    plt.show()
    
    print ("Substract Median Images Complete!")
#End reduceMedianImageStack function  
    

"""
reduceStandardImageStack function takes mean stacked absorber and mean stacked
non-absorber images as parameters. Subtracts noise out by finding difference
between the two. Writes reduced image to new fits file and makes .png image
with Gaussian blur applied.
"""
def reduceStandardImageStack(imageAbsorb, imageNonAbsorb):
    absorber = fits.getdata(imageAbsorb)
    nonAbsorber = fits.getdata(imageNonAbsorb)
    reduced = absorber - nonAbsorber
    fits.writeto('Reduced_Standard.fits', reduced, overwrite=True)
    
    plt.clf()
    plt.imshow(reduced)
    plt.savefig('Reduced_Standard', dpi=100)
    plt.subplots_adjust(right=2.0)
    plt.subplots_adjust(top=1.0)
    plt.colorbar()
    plt.show()

    #Gaussian Blur
    plt.clf()
    betterImage = ndimage.gaussian_filter(reduced, sigma=(2,2), order=0)
    plt.imshow(betterImage)
    plt.savefig('Reduced_Standard_Blur', dpi=100)
    plt.subplots_adjust(right=2.0)
    plt.subplots_adjust(top=1.0)
    plt.colorbar()
    plt.show()
    
    print ("Substract Standard Images Complete!")
#End reduceStandardImageStack function  
    

def brightnessProfile(imageName):
    image = fits.open(imageName)
    newData, header = image[0].data, image[0].header
    print (header)
    height = header['NAXIS2']
    width = header['NAXIS1']
    image.close()
    
    #Begin placing aperture and annulus
    xAxis = (width) / 2             #Center point X-axis
    yAxis = height / 2              #Center point Y-axis
    
    positions = [(xAxis, yAxis)]    #Center point plotted
    aperture = CircularAperture(positions, r=1)
    phot_table_ap = aperture_photometry(newData, aperture)
    r_in = np.linspace(1, 11, 110)   #Inner radii per pixel
    r_out = np.linspace(2, 12, 120)  #Outer radii per pixel
    
    fluxArray = [phot_table_ap['aperture_sum'] / aperture.area()]
    radArray = [1]
    for i in range(0, len(r_in)):
        rIn = r_in[i]
        rOut = r_out[i]
        annulus_apertures = CircularAnnulus(positions, rIn, rOut)
        phot_table = aperture_photometry(newData, annulus_apertures)
        fluxArray.append(phot_table['aperture_sum'] / annulus_apertures.area())
        rMean = (rOut + rIn) / 2
        radArray.append(rMean)
    sumFlux = sum(fluxArray)

    #Plot flux as radius increases
    plt.clf()
    plt.plot(radArray, fluxArray)
    plt.show()     

    fluxFrac = []
    percentSum = 0
    for f in fluxArray:
        percentSum += f
        fluxFrac.append(percentSum / sumFlux)
    
    #Find the radius at 50% enclosed
    radius50 = 1.0
    for i in range(0, len(fluxFrac)):
        if (fluxFrac[i] < .50):
            radius50 += 1/10       #Radius 50% enclosed
    print ("50 percent enclosed:", radius50)
    
    #Find the radius at 90% encolsed
    radius90 = 1.0
    for i in range(0, len(fluxFrac)):
        if (fluxFrac[i] < .90):
            radius90 += 1/10       #Radius 90% enclosed
    print ("90 percent enclosed:", radius90)
    
    """
    #Gaussian Blur
    imgBlur = ndimage.gaussian_filter(newData, sigma=(2,2), order=0)
    plt.imshow(imgBlur)
    plt.savefig('Annuli_' + imageName + '.png', dpi=100)
    plt.subplots_adjust(right=2.0)
    plt.subplots_adjust(top=1.0)
    plt.show()
    """
    
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    circle50 = plt.Circle((xAxis, yAxis), radius=radius50, color='m', fill=False, lw=2)
    circle90 = plt.Circle((xAxis, yAxis), radius=radius90, color='r', fill=False, lw=2)
    plt.imshow(newData, cmap='gray')
    ax.add_patch(circle50) #50% enclosed
    ax.add_patch(circle90) #90% enclosed
    #ax.text(left, top, galRedshift, horizontalalignment='left', verticalalignment='top', fontsize=12, color='red')
    #ax.text(left, bottom, hFlux, horizontalalignment='left', verticalalignment='bottom', fontsize=14, color='green')
    plt.savefig('Annuli_' + imageName + '.png', dpi=100)
    plt.show()     
#End brightnessProfile function
    
    
# =============================================================================
# Read in field absorber files for plots with selected columns
# =============================================================================
fieldList = [1, 2, 3, 4, 5, 7, 8, 9]
absorberObjID = []
absorberRedshift = []
nonAbsorberObjID = []
wavelengthAbsorb = []
nonAbsorberRedshift = []
wavelengthNonAbsorb = []
for i in range(0, len(fieldList)):
    try:
        fileAbsorb = ascii.read('Field' + str(fieldList[i]) + '_Stack_Data_Absorb.dat', delimiter='|')
        absorbObjID = fileAbsorb['col2']
        absorbObjRedshift = fileAbsorb['col3']
        absorbWavelength = fileAbsorb['col4']
        
        for i in range(1, len(absorbObjID)):     
            absorberObjID.append(absorbObjID[i])
            wavelengthAbsorb.append(absorbWavelength[i])
            absorberRedshift.append(float(absorbObjRedshift[i]))
            
    except IOError:
        print ("File not found!")

for i in range(0, len(fieldList)):
    try:        
        fileNonAbsorb = ascii.read('Field' + str(fieldList[i]) + '_Stack_Data_NonAbsorb.dat', delimiter='|')
        nonAbsorbObjID = fileNonAbsorb['col2']
        nonAbsorbObjRedshift = fileNonAbsorb['col3']
        nonAbsorbWavelength = fileNonAbsorb['col4']
        
        for i in range(1, len(nonAbsorbObjID)):
            nonAbsorberObjID.append(nonAbsorbObjID[i])
            wavelengthNonAbsorb.append(nonAbsorbWavelength[i])
            nonAbsorberRedshift.append(float(nonAbsorbObjRedshift[i]))
    
    except IOError:
        print ("File not found!")
    
#print (absorberObjID,'\n')
print (absorberRedshift,'\n')
#print (nonAbsorberObjID,'\n')
print (nonAbsorberRedshift,'\n')
    
# =============================================================================
# Program's 'main'. Begin reading in all images for stacking.
# =============================================================================
"""
Define list containing field numbers, go through list and read in fields. Call
stack function to stack all fields. 
"""
fileListAll = []
fileListMeanAll = []
fileListMedianAll = []
fileListMeanAbsorb = []
fileListMedianAbsorb = []
fileListMeanNonAbsorb = []
fileListMedianNonAbsorb = []
fileListStandardAbsorb = []
fileListStandardNonAbsorb = []
galList = [1, 2, 3, 4, 5, 7, 8, 9]

for i in range(0, len(galList)):
    try:
        fileNameAll = 'Field' + str(galList[i]) + '_Stacked_Image_Normed_All.fits'
        fileMeanAll = 'Field' + str(galList[i]) + '_Stacked_Image_Mean_Normed_All.fits'
        fileMedianAll = 'Field' + str(galList[i]) + '_Stacked_Image_Median_Normed_All.fits'
        fileMeanAbsorb = 'Field' + str(galList[i]) + '_Stacked_Image_Mean_Normed_Absorber.fits'
        fileNameStandAbsorb = 'Field' + str(galList[i]) + '_Stacked_Image_Normed_Absorber.fits'
        fileNameStandNonAbsorb = 'Field' + str(galList[i]) + '_Stacked_Image_Normed_NonAbsorber.fits'
        fileNameMedianAbsorb = 'Field' + str(galList[i]) + '_Stacked_Image_Median_Normed_Absorber.fits'
        fileNameMeanNonAbsorb = 'Field' + str(galList[i]) + '_Stacked_Image_Mean_Normed_NonAbsorber.fits'
        fileNameMedianNonAbsorb = 'Field' + str(galList[i]) + '_Stacked_Image_Median_Normed_NonAbsorber.fits'
                 
        #Append normalized image to fileList to pass as argument to stack function.
        fileListAll.append(fileNameAll)
        fileListMeanAll.append(fileMeanAll)
        fileListMedianAll.append(fileMedianAll)
        fileListMeanAbsorb.append(fileMeanAbsorb)
        fileListMedianAbsorb.append(fileNameMedianAbsorb)
        fileListMeanNonAbsorb.append(fileNameMeanNonAbsorb)
        fileListMedianNonAbsorb.append(fileNameMedianNonAbsorb)
        fileListStandardAbsorb.append(fileNameStandAbsorb)
        fileListStandardNonAbsorb.append(fileNameStandNonAbsorb)
        
    except IOError:
        print ("Image ID " + str(galList[i]) + " not found!")

print (len(fileListAll))
print (len(fileListMeanAll))
print (len(fileListMedianAll))
print (len(fileListMeanAbsorb))
print (len(fileListMedianAbsorb))
print (len(fileListMeanNonAbsorb))
print (len(fileListMedianNonAbsorb))
print (len(fileListStandardAbsorb))
print (len(fileListStandardNonAbsorb))
    
#Call Stack functions
stackAll(fileListAll)
stackMeanAll(fileListMeanAll)
stackMedianAll(fileListMedianAll)
meanAbsorbImage = stackMeanAbsorb(fileListMeanAbsorb)
stackMedianAbsorb(fileListMedianAbsorb)
meanNonAbsorbImage = stackMeanNonAbsorb(fileListMeanNonAbsorb)
stackMedianNonAbsorb(fileListMedianNonAbsorb)
stackStandardAbsorb(fileListStandardAbsorb)
stackStandardNonAbsorb(fileListStandardNonAbsorb)
#End program 'main' section

# =============================================================================
# Read in combined image stacks and pass as parameters to reduction functions.
# =============================================================================
reduceMeanImageStack('Stacked_Image_Mean_Absorb.fits', 'Stacked_Image_Mean_NonAbsorb.fits')
reduceMedianImageStack('Stacked_Image_Median_Absorb.fits', 'Stacked_Image_Median_NonAbsorb.fits')
reduceStandardImageStack('Stacked_Image_Standard_Absorb.fits', 'Stacked_Image_Standard_NonAbsorb.fits')

# =============================================================================
# Measure brightness profile of stacked images.
# =============================================================================
brightnessProfile('Stacked_Image_All.fits')
brightnessProfile('Stacked_Image_Mean_Absorb.fits')
brightnessProfile('Stacked_Image_Mean_NonAbsorb.fits')
brightnessProfile('Stacked_Image_Mean_All.fits')
brightnessProfile('Stacked_Image_Median_Absorb.fits')
brightnessProfile('Stacked_Image_Median_Nonabsorb.fits')
brightnessProfile('Stacked_Image_Median_All.fits')
brightnessProfile('Stacked_Image_Standard_Absorb.fits')
brightnessProfile('Stacked_Image_Standard_Nonabsorb.fits')

# =============================================================================
# Begin section for ks statistics and plots.
# =============================================================================
print ("---------------------------------------------------------------------------")
print ("Statistics",'\n')
"""
Perform statistics on absorber redshifts vs. non-absorber redshifts.
"""       
print ("Redshifts Absorber/Non-Absorber")
print (stats.ks_2samp(absorberRedshift, nonAbsorberRedshift),'\n')

print ("Mean Stacks Absorber/Non-Absorber")
print (stats.ks_2samp(np.concatenate(meanAbsorbImage), np.concatenate(meanNonAbsorbImage)))
 
print("---------------------------------------------------------------------------")

print ("Absorber:", absorberRedshift,'\n')
print ("Non-Absorber:", nonAbsorberRedshift)
print ("Mean Absorber:", np.mean(absorberRedshift))
print ("Median Absorber:", np.median(absorberRedshift))
print ("Mean non-absorber:", np.mean(nonAbsorberRedshift))
print ("Median non-absorber:", np.median(nonAbsorberRedshift))
print ("stdv mean:", np.std(absorberRedshift))

"""
Histogram plots for absorber/non-absorber redshifts.
"""
absorberBinArray = np.linspace(.65, 1.7, 10)
nonAbsorberBinArray = np.linspace(.65, 1.7, 10)
plt.hist(absorberRedshift, bins=absorberBinArray, density=True, histtype='step', label='MgII Detection (%i)' %len(absorberRedshift))
plt.hist(nonAbsorberRedshift, bins=nonAbsorberBinArray,  density=True, histtype='step', label='MgII Detection (%i)' %len(nonAbsorberRedshift))   
plt.legend()
plt.savefig('Redshift Histogram.png')
plt.show()    
    

