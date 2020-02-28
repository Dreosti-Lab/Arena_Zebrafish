# -*- coding: utf-8 -*-
"""
Created on Mon Oct 28 14:52:52 2019

@author: Tom Ryan (Dreosti Lab, UCL)
Adapted from Social Zebrafish library by Dreosti-Lab
"""
# Import useful libraries
import numpy as np
import matplotlib.pyplot as plt
import scipy.misc as misc
import math
import cv2
import timeit

# Process Video : Track fish in AVI
def arena_fish_tracking(aviFile, output_folder, ROI, plot):

    d,expName=aviFile.rsplit('\\',1)  # take last part of aviFile path
    expName=expName[0:-4]    
    # Load Video
    vid = cv2.VideoCapture(aviFile)
    # find size of ROI, or whole image if no ROIs
    if(len(ROI)==0):
        w = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
        ROI=[[0,0,w,h]]
        ROI=np.asarray(ROI)
    else:
        w, h = get_ROI_size(ROI, 0)
        
    # Compute a "Starting" Background
    # - Median value of 20 frames with significant difference between them
    
    background = compute_initial_background(vid, ROI)
    # 5 seconds
    
    # Algorithm
    # 1. Find initial background guess for the ROI
    # 2. Extract Crop regions from the ROI
    # 3. Threshold ROI using median/7 of each crop region, Binary Close image using 5 rad disc
    # 4. Find largest particle (Contour)
    # 5. - Compute Weighted Centroid (X,Y) for Eye Region (10% of brightest pixels)
    # 6. - Compute Binary Centroid of Body Region (50% of brightest pixels - eyeRegion)
    # 7. - Compute Heading
     
    vid = cv2.VideoCapture(aviFile)  
    numFrames = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))-100 # Skip, possibly corrupt, last 100 frames (1 second), and the first 30 seconds
    startFrame = 0
    previous_ROI=[]
    previous_ROI.append(np.zeros((h,w),dtype = np.uint8))
    
    # Allocate Tracking Data Space
    fxS = np.zeros((numFrames,1))           # Fish X
    fyS = np.zeros((numFrames,1))           # Fish Y
    bxS = np.zeros((numFrames,1))           # Body X
    byS = np.zeros((numFrames,1))           # Body Y
    exS = np.zeros((numFrames,1))           # Eye X
    eyS = np.zeros((numFrames,1))           # Eye Y
    areaS = np.zeros((numFrames,1))         # area (-1 if error)
    ortS = np.zeros((numFrames,1))          # heading/orientation (angle from body to eyes)
    motS = np.zeros((numFrames,1))          # frame-by-frame change in segmented particle
       
#    numFrames=10*120
#    For testing
    vid.set(cv2.CAP_PROP_POS_FRAMES, startFrame)    # start at startFrame
    for f in range(startFrame,numFrames):
        
        # Report Progress every 120 frames of movie
#        if (f%120) == 0:
#            print ('\r'+ str(f) + ' of ' + str(numFrames) + ' frames done')
        # Read next frame        
        ret, im = vid.read()
        
        # Convert to grayscale (uint8)
        current = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        crop, xOff, yOff = get_ROI_crop(current, ROI,0)
        diffimg= background - crop                
        diffimg[diffimg>220] = 0  #remove nearly saturated differences (not sure where these come from... water evaporation and vibration from drilling maybe?)
        # 0.005 s
        
        # Determine current threshold
        threshold_level = np.median(background)/5           
        
        # Threshold            
        level, threshold = cv2.threshold(diffimg,threshold_level,255,cv2.THRESH_BINARY)
        threshold = np.uint8(diffimg > threshold_level)
         
        # Binary Close
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
        closing = cv2.morphologyEx(threshold, cv2.MORPH_CLOSE, kernel)
            
        # Find Binary Contours            
        contours, hierarchy = cv2.findContours(closing,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
        # takes virtually no time
               
        if f%1200 == 0:
            print(f)
            
        # Create Binary Mask Image
        mask = np.zeros(crop.shape,np.uint8)
                       
        # If there are NO contours, then skip tracking
        if len(contours) == 0:
            print('No Contour')
            if f!= 0:
                area = -1.0
                fX = fxS[f-1] - xOff
                fY = fyS[f-1] - yOff
                bX = bxS[f-1] - xOff
                bY = byS[f-1] - yOff
                eX = exS[f-1] - xOff
                eY = eyS[f-1] - yOff
                heading = ortS[f-1]
                motion = -1.0
            else:
                area = -1.0
                fX = xOff
                fY = yOff
                bX = xOff
                bY = yOff
                eX = xOff
                eY = yOff
                heading = -181.0
                motion = -1.0
        
        else:
            
            # Get Largest Contour (fish, ideally)
            largest_cnt, area = get_largest_contour(contours)
            
            # If the particle to too small to consider, skip frame
            if area == 0.0:
                print('Particle too small')
                if f!= 0:
                    fX = fxS[f-1] - xOff
                    fY = fyS[f-1] - yOff
                    bX = bxS[f-1] - xOff
                    bY = byS[f-1] - yOff
                    eX = exS[f-1] - xOff
                    eY = eyS[f-1] - yOff
                    heading = ortS[f-1]
                    motion = -1.0
                else:
                    area = -1.0
                    fX = xOff
                    fY = yOff
                    bX = xOff
                    bY = yOff
                    eX = xOff
                    eY = yOff
                    heading = -181.0
                    motion = -1.0
                    
            else:
                # Draw contours into Mask Image (1 for Fish, 0 for Background
                cv2.drawContours(mask,[largest_cnt],0,1,-1) # -1 draw the contour filled
                pixelpoints = np.transpose(np.nonzero(mask))
                
                # Get Area (again)
                area = np.size(pixelpoints, 0)
                
# ---------------------------------------------------------------------------------
                # Compute Frame-by-Frame Motion (absolute changes above threshold)
                # - Normalize by total absdiff from background
                
                if (f != 0):
                    #absdiff = cv2.absdiff(background, crop)
                    absdiff = np.abs(diffimg)
                    absdiff[absdiff < threshold_level] = 0
                    totalAbsDiff = np.sum(absdiff)
                    frame_by_frame_absdiff = np.abs(np.float32(previous_ROI) - np.float32(crop)) / 2 # Adjust for increases and decreases across frames
                    frame_by_frame_absdiff[frame_by_frame_absdiff < threshold_level] = 0
                    motion = np.sum(np.abs(frame_by_frame_absdiff))/totalAbsDiff
                    # 0.01 s
                else:
                    motion = 0
                    
                    # Save Masked Fish Image from ROI (for subsequent frames motion calculation)
                    previous_ROI = np.copy(crop)
                #
                # ---------------------------------------------------------------------------------
                # Find Body and Eye Centroids
                area = np.float(area)
                
                # Highlight 50% of the birghtest pixels (body + eyes)                    
                numBodyPixels = np.int(np.ceil(area/2))
                    
                # Highlight 10% of the birghtest pixels (mostly eyes)     
                numEyePixels = np.int(np.ceil(area/10))
                    
                # Fish Pixel Values (difference from background)
                fishValues = diffimg[pixelpoints[:,0], pixelpoints[:,1]]
                sortedFishValues = np.sort(fishValues)
                
                bodyThreshold = sortedFishValues[-numBodyPixels]                    
                eyeThreshold = sortedFishValues[-numEyePixels]

                # Compute Binary/Weighted Centroids
                r = pixelpoints[:,0]
                c = pixelpoints[:,1]
                all_values = diffimg[r,c]
                all_values = all_values.astype(float)
                r = r.astype(float)
                c = c.astype(float)
                
                # Fish Centroid
                values = np.copy(all_values)
                values = (values-threshold_level+1)
                acc = np.sum(values)
                fX = np.float(np.sum(c*values))/acc
                fY = np.float(np.sum(r*values))/acc
                
                # Eye Centroid (a weighted centorid)
                values = np.copy(all_values)                   
                values = (values-eyeThreshold+1)
                values[values < 0] = 0
                acc = np.sum(values)
                eX = np.float(np.sum(c*values))/acc
                eY = np.float(np.sum(r*values))/acc

                # Body Centroid (a binary centroid, excluding "eye" pixels)
                values = np.copy(all_values)                   
                values[values < bodyThreshold] = 0
                values[values >= bodyThreshold] = 1                                                            
                values[values > eyeThreshold] = 0                                                            
                acc = np.sum(values)
                bX = np.float(np.sum(c*values))/acc
                bY = np.float(np.sum(r*values))/acc
                
                # ---------------------------------------------------------------------------------
                # Heading (0 deg to right, 90 deg up)
                if (bY != eY) or (eX != bX):
                    heading = math.atan2((bY-eY), (eX-bX)) * (360.0/(2*np.pi))
                else:
                    heading = -181.00
                # entire module above takes 0.025 s for first frame, then ~0.05 every other frame... why longer for next frames?  
        
        # ---------------------------------------------------------------------------------
        # Store data in arrays
        # Shift X,Y Values by ROI offset and store in Matrix
        
        
        fxS[f,0] = fX + xOff
        fyS[f,0] = fY + yOff
        bxS[f,0] = bX + xOff
        byS[f,0] = bY + yOff
        exS[f,0] = eX + xOff
        eyS[f,0] = eY + yOff
        areaS[f,0] = area
        ortS[f,0] = heading
        motS[f,0] = motion
                
        # -----------------------------------------------------------------
        # Update this ROIs background estimate (everywhere except the (dilated) Fish)
        current_background = np.copy(background)            
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(15,15))
        dilated_fish = cv2.dilate(mask, kernel, iterations = 2)           
        updated_background = (np.float32(crop) * 0.01) + (current_background * 0.99)
        updated_background[dilated_fish==1] = current_background[dilated_fish==1]            
        background = np.copy(updated_background)
        # 0.02s every frame
        # ---------------------------------------------------------------------------------
        # Plot Fish in Movie with Tracking Overlay?
        if(plot==1):
            if (f == 0) or (f == numFrames-1):
                print(f)
                plt.clf()
                enhanced = cv2.multiply(current, 1)
                color = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
                plt.imshow(color)
                plt.axis('image')
                plt.plot(fxS[:,0],fyS[:,0],'b.', MarkerSize = 1)
                plt.plot(exS[:,0],eyS[:,0],'r.', MarkerSize = 1)
                plt.plot(bxS[:,0],byS[:,0],'co', MarkerSize = 1)
                if (f % 1000 == 0):
                    plt.text(bxS[f,0]+10,byS[f,0]+10,  '{0:.1f}'.format(ortS[f,0]), color = [1.0, 1.0, 0.0, 0.5])
                    plt.text(bxS[f,0]+10,byS[f,0]+30,  '{0:.0f}'.format(areaS[f,0]), color = [1.0, 0.5, 0.0, 0.5])
                    plt.draw()
                    plt.pause(0.001)
                    # plotting this takes 0.5s even for a single point - minimize!!
# ---------------------------------------------------------------------------------
# Save Tracking Summary
        if(f == 0):
#            print('Making tracking image')
            plt.savefig(output_folder + '\\' + expName + '_initial_tracking.png', dpi=300)
            plt.figure('backgrounds')
            plt.imshow(background)
            plt.savefig(output_folder+'\\' + expName +'_initial_background.png', dpi=300)
            plt.close('backgrounds')
       
        if(f == numFrames-1):
#            print('Adding to tracking image')
            plt.savefig(output_folder+'\\' + expName +'_final_tracking.png', dpi=300)
            plt.figure('backgrounds')
            plt.imshow(background)
            plt.savefig(output_folder+'\\' + expName +'_final_background.png', dpi=300)
            plt.close('backgrounds')
            
# -------------------------------------------------------------------------
# Close Video File
    vid.release()

# Return tracking data
    return fxS, fyS, bxS, byS, exS, eyS, areaS, ortS, motS
#------------------------------------------------------------------------------
    

# Return ROI size from ROI list
def get_ROI_size(ROIs, numROi):
    width = np.int(ROIs[numROi, 2])
    height = np.int(ROIs[numROi, 3])
    
    return width, height


def compute_initial_background(vid, ROI):

    numFrames = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))-100 # Skip, possibly corrupt, last 100 frames (1 second)
    # Allocate space for ROI background
    background_ROI = []
    w, h = get_ROI_size(ROI, 0)
    background_ROI.append(np.zeros((h, w), dtype = np.float32))
    
    # Find initial background for the ROI
    crop_width, crop_height = get_ROI_size(ROI, 0)
    bFrames = 100
    stepFrames = np.int(np.floor_divide(np.floor(numFrames*0.05),bFrames)) # Check background frame uniformly across time series for [numSteps] frames
    #stepFrames = 360 # Check background frame every 3 seconds
    
    backgroundStack = np.zeros((crop_height, crop_width, bFrames), dtype = np.float32)  
    previous = np.zeros((crop_height, crop_width), dtype = np.float32)

    # Store first frame
    vid.set(cv2.CAP_PROP_POS_FRAMES, 200) # skip first 200 frames
    ret, im = vid.read()
    current = np.float32(cv2.cvtColor(im, cv2.COLOR_BGR2GRAY))
    crop, xOff, yOff = get_ROI_crop(current, ROI,0)
    backgroundStack[:,:,0] = np.copy(crop)
    previous = np.copy(crop)
    bCount = 0
    
        # Search for useful background frames every stepFrames (can add significantly different than previous functionality)
    changes = []
    for f in range(stepFrames, numFrames, stepFrames):
        # Read frame
        vid.set(cv2.CAP_PROP_POS_FRAMES, f)
        ret, im = vid.read()
        current = np.float32(cv2.cvtColor(im, cv2.COLOR_BGR2GRAY))
        crop, xOff, yOff = get_ROI_crop(current, ROI, 0)
    
        # Measure change from current to previous frame
        absdiff = np.abs(previous-crop)
        level = np.median(crop)/5
        change = np.mean(absdiff > level)
        changes.append(change)
        previous = np.copy(crop)
        #print(change)
        # If significant, add to stack...possible finish
        if(change > 0):
            backgroundStack[:,:,bCount] = np.copy(crop)
            bCount = bCount + 1
            #print(bCount)
            if(bCount == bFrames):
                print("Background for ROI found on frame " + str(f))
                break
    
    # Compute background
    print('collapsing')
    backgroundStack = backgroundStack[:,:, 0:bCount]
    background_ROI = np.uint8(np.median(backgroundStack, axis=2))
                  
    # Return initial background
    return background_ROI

def get_ROI_crop(image, ROIs, numROi):
    r1 = np.int(ROIs[numROi, 1])
    r2 = np.int(r1+ROIs[numROi, 3])
    c1 = np.int(ROIs[numROi, 0])
    c2 = np.int(c1+ROIs[numROi, 2])
    crop = image[r1:r2, c1:c2]
    
    return crop, c1, r1

# Return largest (area) cotour from contour list
def get_largest_contour(contours):
    # Find contour with maximum area and store it as best_cnt
    max_area = 0
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > max_area:
            max_area = area
            best_cnt = cnt
    if max_area > 0:
        return best_cnt, max_area
    else:
        return cnt, max_area

# Compare Old (scaled) and New (non-scaled) background images
def compare_backgrounds(folder):

    # Load -Initial- Background Frame (histogram from first 50 seconds)
    backgroundFile = folder + r'/background_old.png'
    background_old = misc.imread(backgroundFile, False)
    
    backgroundFile = folder + r'/background.png'
    background = misc.imread(backgroundFile, False)
    absDiff = cv2.absdiff(background_old, background)

    return np.mean(absDiff)

# FIN