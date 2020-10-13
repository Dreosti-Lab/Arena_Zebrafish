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
import AZ_utilities as AZU

# Process Video : Track fish in AVI
def arena_fish_tracking(aviFile, output_folder, ROI,plot=1,cropOp=1,FPS=120,saveCroppedMovie=1,cropSize=[128,128],startFrame=0):

    #    l=0
    
    if plot==1:
        pmsg='ON'
    else:
        pmsg='OFF'
    if cropOp==1:
        cmsg='ON'
    else:
        cmsg='OFF. WARNING!! Tracking can take a very long time with cropping turned OFF'
    if saveCroppedMovie==1:
        smsg='ON. WARNING!! this takes some time'
    else:
        smsg='OFF'
        
    message='Tracking fish with plotting turned ' + pmsg + ' and cropping turned ' + cmsg + '. Cropped movie generation is ' + smsg
    print(message)
    
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
    OrigROI=np.copy(ROI)
    
    failedAviFiles=False
    # Compute a "Starting" Background
    # - Median value of 100 frames sampled evenly across first 30 secs
    
    backgroundFull = compute_initial_background(vid, ROI)
    
    # 5 seconds
    
    # Algorithm
    # 1. Find initial background guess for the whole image
    # 2. Find difference of startFrame and initial background
    # 3. Threshold image using median/7 of each crop region, Binary Close image using 5 rad disc
    # 4. Find largest particle (Contour)
    # 5. - Compute Weighted Centroid (X,Y) for Eye Region (10% of brightest pixels)
    # 6. - Compute Binary Centroid of Body Region (50% of brightest pixels - eyeRegion)
    # 7. - Compute Heading
    # 8. Dilate fish and update background (whole image)
    # 9. Find crop region based on fx,fy coordinates and provided cropSize
    # 10. Crop previous_ROI, current, and background on following loops (steps 3 through 7)
    # 11. Save cropped movie
    #maxNumFrames=432000
    numFrames = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))-100 # Skip, possibly corrupt, last 100 frames (1 second), and the first 30 seconds
    #if numFrames>maxNumFrames:numFrames=maxNumFrames
    
    numFrames=numFrames-startFrame
    previous_ROI=[]
    previous_ROI.append(np.zeros((h,w),dtype = np.uint8))

#    For testing
    #numFrames=30*120
    ########################
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
      
    cropFlag = 0
    vid.set(cv2.CAP_PROP_POS_FRAMES, startFrame)    # start at startFrame
    print('Tracking')
    
    noContoursCount=0               
    limitError=0.1
    limitFrames=np.floor(numFrames*limitError)
    for f in range(numFrames):
#        print(f)
#        if(f==4157):
#            print('here')
#        # Report Progress every 120 frames of movie
#        if (f%120) == 0:
#            print ('\r'+ str(f) + ' of ' + str(numFrames) + ' frames done')
        
        
        ################### TESTING ######################
#        if(f==72016):
#            print('here')
#        print(f)
#        print(f+startFrame)
        ##################################################
        
        # Read next frame 
        ret, im = vid.read()
        errF=-1
        
        # check we haven't lost the fish (more than 10% of movie continuously)
        # return error frame and break the loop
        if(noContoursCount>limitFrames):
            failedAviFiles=True
            errF=f-noContoursCount
            break
        
        # Convert to grayscale (uint8)
        current = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        crop, xOff, yOff = get_ROI_crop(current, ROI,0)
        background, xOff, yOff = get_ROI_crop(backgroundFull, ROI,0)                      
        diffimg= background - crop                
        diffimg[diffimg>220] = 0  #remove nearly saturated differences; at edges of objects (not sure where these come from... water evaporation and vibration from drilling maybe?)
            # 0.005 s
            
        # Determine current threshold
        # Because the edges are much lower in intensity, this can mess with the threshold if too much of the image is black
        # therefore we only include values above 30 (lowest measured pixel value of navigable chamber at set IR intensity)
        threshold_level = np.median(background[background>30])/6     
        
        # Threshold            
        level, threshold = cv2.threshold(diffimg,threshold_level,255,cv2.THRESH_BINARY)
        threshold = np.uint8(diffimg > threshold_level)
         
        # Binary Close
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
        closing = cv2.morphologyEx(threshold, cv2.MORPH_CLOSE, kernel)
            
        # Find Binary Contours            
        contours, hierarchy = cv2.findContours(closing,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
        # takes virtually no time
            
        # Create Binary Mask Image
        mask = np.zeros(crop.shape,np.uint8)
        
        # If there are NO contours, then skip tracking
        if len(contours) == 0:
            print('No Contour')
            noContoursCount+=1
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
            noContoursCount=0   # Reset contour count if we found the fish again
            # Get Largest Contour (fish, ideally)
            largest_cnt, area = get_largest_contour(contours,mask,area_limit=285)
            
            # If the particle to too small to consider, skip frame
            if area == 0.0:
                print('Particle too small')
                noContoursCount+=1
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
                pixels = np.nonzero(mask)
                
                #Check the contour isn't a ridiculous shape
                pixelpoints = np.transpose(pixels)
                # Get Area (again)
                area = np.size(pixelpoints, 0)
                
# ---------------------------------------------------------------------------------
                # Compute Frame-by-Frame Motion (absolute changes above threshold)
                # - Normalize by total absdiff from background
                
                if (f!= 0):
                    absdiff = np.abs(diffimg)
                    absdiff[absdiff < threshold_level] = 0
                    totalAbsDiff = np.sum(absdiff)
                    frame_by_frame_absdiff = np.abs(np.float32(previous_ROI) - np.float32(crop)) / 2 # Adjust for increases and decreases across frames
                    frame_by_frame_absdiff[frame_by_frame_absdiff < threshold_level] = 0
                    motion = np.sum(np.abs(frame_by_frame_absdiff))/totalAbsDiff
                    # 0.01 s
                else:
                    motion = 0
                 
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
                # Now crop the movie around the fish
                # crop movie here and recompute diffimg if working with the first frame
                # find start and end positions for cropping around the located fish
                crop,startIdX,startIdY,endIdX,endIdY=cropImFromTracking(vid,current,fX + xOff,fY + yOff,cropSize)
                ROI=np.asarray([[startIdX,startIdY,cropSize[0],cropSize[1]]])
                # If this is the first time, then we have found the fish on the whole image. Will work with cropped images from now on
                if(cropFlag==0):
                    mask=mask[startIdY:endIdY,startIdX:endIdX]
                    backgroundFull=np.copy(background)
                    background=background[startIdY:endIdY,startIdX:endIdX]
                    cropFlag = 1
            
        previous_ROI = np.copy(crop)
#        if(saveCroppedMovie==1):
#            croppedMovie.append(crop)
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
        # Update the whole background estimate (everywhere except the (dilated) Fish)
        # Every 2 mins, recompute complete background
        backgroundInterval=int(FPS*120)
       
        if((f%(backgroundInterval)==0) & (f!=0)):
            current_background=np.copy(backgroundFull)
            backgroundFull,_=recomputeBackground(vid,f+startFrame,OrigROI)
            
            # Figure out where fish pixels are in the full image: hideous coding but the ndarray is messy to play with
            largest_cntFull=np.copy(largest_cnt)
            for i in range(len(largest_cnt)): # cycle through pixels
                largest_cntFull[i][0][0]+=xOff
                largest_cntFull[i][0][1]+=yOff
            
            # Draw the fish onto maskFull
            maskFull=np.zeros(current.shape)
            cv2.drawContours(maskFull,[largest_cntFull],0,1,-1) # -1 draw the contour filled
            pixels = np.nonzero(maskFull)
            # Dilate the fish and exclude it from the updated background
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(10,10))
            dilated_fish = cv2.dilate(maskFull, kernel, iterations = 2)                       
            backgroundFull[dilated_fish==1] = current_background[dilated_fish==1]
            
            vid.set(cv2.CAP_PROP_POS_FRAMES,f+startFrame) # reset frame number
            print('continuing tracking')            
            
#        updated_background = (np.float32(crop) * 0.01) + (current_background * 0.99)
        # 0.02s every frame
        # ---------------------------------------------------------------------------------
        # Plot Fish in Movie with Tracking Overlay?
        if(plot==1):
            if(f==0) or (f == numFrames-1):
#            if (f%30==0): # every 5 seconds
#                print(f)
                plt.clf()
                enhanced = cv2.multiply(current, 1)
                color = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
                plt.imshow(color)
                plt.axis('image')
                plt.plot(fxS[:,0],fyS[:,0],'b.', MarkerSize = 1)
                plt.plot(exS[:,0],eyS[:,0],'r.', MarkerSize = 1)
                plt.plot(bxS[:,0],byS[:,0],'co', MarkerSize = 1)
#                if(l<10):
#                    ll='00'+str(l)
#                else:
#                    if(l<100):
#                        ll='0'+str(l)
#                    else:
#                        ll=str(l)
#                        
#                plt.savefig(output_folder + '\\' + ll + '.png', dpi=300)
#                l+=1
#                if (f % 7200 == 0): # every minute
#                    plt.text(bxS[f,0]+10,byS[f,0]+10,  '{0:.1f}'.format(ortS[f,0]), color = [1.0, 1.0, 0.0, 0.5])
#                    plt.text(bxS[f,0]+10,byS[f,0]+30,  '{0:.0f}'.format(areaS[f,0]), color = [1.0, 0.5, 0.0, 0.5])
#                    plt.draw()
#                    plt.pause(0.001)
        else:  
            if (f == 0) or (f == numFrames-1): # only plot this in the first and last frame to save the file
                
                plt.clf()
                enhanced = cv2.multiply(current, 1)
                color = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
                plt.imshow(color)
                plt.axis('image')
                plt.plot(fxS[:,0],fyS[:,0],'b.', MarkerSize = 1)
                plt.plot(exS[:,0],eyS[:,0],'r.', MarkerSize = 1)
                plt.plot(bxS[:,0],byS[:,0],'co', MarkerSize = 1)
#                if (f % 1000 == 0):
#                    plt.text(bxS[f,0]+10,byS[f,0]+10,  '{0:.1f}'.format(ortS[f,0]), color = [1.0, 1.0, 0.0, 0.5])
#                    plt.text(bxS[f,0]+10,byS[f,0]+30,  '{0:.0f}'.format(areaS[f,0]), color = [1.0, 0.5, 0.0, 0.5])
#                    plt.draw()
#                    plt.pause(0.001)
#                    
# ---------------------------------------------------------------------------------
# Save Tracking Summary
        if(f == 0):
#            print('Making tracking image')
            path=output_folder + '\\' + expName + '_initial_tracking.png'
            plt.savefig(path, dpi=300)
#            AZU.createShortcutTele(path)
            
            plt.figure('backgrounds')
            plt.imshow(backgroundFull)
            path=output_folder+'\\' + expName +'_initial_background.png'
            plt.savefig(path, dpi=300)
#            AZU.createShortcutTele(path)
            plt.close('backgrounds')
       
        if(f == numFrames-1):
            path=output_folder+'\\' + expName +'_final_tracking.png'
            plt.savefig(path, dpi=300)
#            AZU.createShortcutTele(path)
            
            plt.figure('backgrounds')
            plt.imshow(backgroundFull)
            path=output_folder+'\\' + expName +'_final_background.png'
            plt.savefig(path, dpi=300)
#            AZU.createShortcutTele(path)
            plt.close('backgrounds')
            
        if(f==math.floor(numFrames/2)):
            print('Halfway done tracking')
         
        
        # Save cropped movie (frame by frame for memory efficiency)?     
        if((saveCroppedMovie==1) & (f==0)):
            saveName=output_folder+'\\' + expName +'_cropped.avi'
            # create out write object
            width=cropSize[0]
            height=cropSize[1]
            out = cv2.VideoWriter(saveName, cv2.VideoWriter_fourcc(*'DIVX'), FPS, (width,height), False)
#            AZU.createShortcutTele(saveName)
        if(saveCroppedMovie==1):
            im_crop,_,_,_,_=cropImFromTracking(vid,current,fxS[f],fyS[f],cropSize)
            out.write(im_crop)
            
        ##### END OF FRAME LOOP #####
    # Save cropped movie?     
#    if(saveCroppedMovie==1):
#        saveName=output_folder+'\\' + expName +'_cropped.avi'
        # crop movie from fx
#        saveGrayImgListAsMovie(croppedMovie,saveName,FPS,cropSize)
#        makeCroppedMovieFromTracking(fxS,fyS,vid,saveName,FPS,cropSize)
# -------------------------------------------------------------------------
# Close Video Files
    if(saveCroppedMovie==1):
        out.release()
    
    vid.release()
    print('Finished tracking')
   
    # Return tracking data
    return fxS, fyS, bxS, byS, exS, eyS, areaS, ortS, motS, failedAviFiles, errF

###############################################################################
# Crop a single frame using tracking coordinates and return cropped frame and crop indices    
def cropImFromTracking(vid,im,fx,fy,cropSize):

    if(math.isinf(fx)):
        print('here')
        
    startIdX = math.ceil(fx) - math.ceil(cropSize[0] / 2)
    endIdX = math.ceil(fx) + math.ceil(cropSize[0] / 2)
    startIdY = math.ceil(fy) - math.ceil(cropSize[1] / 2)
    endIdY = math.ceil(fy) + math.ceil(cropSize[1] / 2)
    
    # Check we don't fall off the edge of the image
    # if it does set start and end to min max of image
    wFull = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
    hFull = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    if(startIdY<0):
        endIdY=endIdY+(startIdY*-1)                
        startIdY=0
    if(startIdX<0):
        endIdX=endIdX+(startIdX*-1)
        startIdX=0
    if(endIdX>wFull):
        endIdX=wFull
        startIdX=endIdX-(cropSize[0])
    if(endIdY>hFull):
        endIdY=hFull
        startIdY=endIdY-(cropSize[1])
        
    im_crop=im[startIdY:endIdY,startIdX:endIdX]
    return im_crop,startIdX,startIdY,endIdX,endIdY

##############################################################################
# use tracking data to crop area around a fish and save the movie    
def makeCroppedMovieFromTracking(fxS,fyS,vid,saveName,FPS,cropSize):
    AZU.setFrame(vid,0)
    width=cropSize[0]
    height=cropSize[1]
    out = cv2.VideoWriter(saveName+'.avi',cv2.VideoWriter_fourcc(*'DIVX'), FPS, (width,height), False)
    print('Writing cropped video')
    for i in range(len(fxS)):
        ret, im = vid.read()
        im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        
        startIdX = math.ceil(fxS[i]) - math.ceil(cropSize[0] / 2)
        endIdX = math.ceil(fxS[i]) + math.ceil(cropSize[0] / 2)
        startIdY = math.ceil(fyS[i]) - math.ceil(cropSize[1] / 2)
        endIdY = math.ceil(fyS[i]) + math.ceil(cropSize[1] / 2)
        
        # Check we don't fall off the edge of the image
        # if it does set start and end to min max of image
        wFull = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
        hFull = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        if(startIdY<0):
            endIdY=endIdY+(startIdY*-1)                
            startIdY=0
        if(startIdX<0):
            endIdX=endIdX+(startIdX*-1)
            startIdX=0
        if(endIdX>wFull):
            startIdX=startIdX-(wFull-endIdX)
            endIdX=wFull
        if(endIdY>hFull):
            startIdY=startIdY-(hFull-endIdY)
            endIdY=hFull
            
        im_crop=im[startIdY:endIdY,startIdX:endIdX]
        out.write(im_crop)
    print('Done')

###############################################################################
# Save a listArray as an avi movie
def saveGrayImgListAsMovie(list,saveName,FPS,size): # convenient and very fast but need enough RAM to store this list array
                           
        width=size[0]
        height=size[1]
        out = cv2.VideoWriter(saveName+'.avi',cv2.VideoWriter_fourcc(*'DIVX'), FPS, (width,height), False)
        for i in range(len(list)):
            out.write(list[i])
        out.release()

##############################################################################
# Return ROI size from ROI list
def get_ROI_size(ROIs, numROi):
    width = np.int(ROIs[numROi, 2])
    height = np.int(ROIs[numROi, 3])
    
    return width, height

def  recomputeBackground(vid,f,ROI):
    
    print('Updating background (' + str((f/120)/60) + ' mins done)')
    # Allocate space for ROI background
    background_ROI = []
    w, h = get_ROI_size(ROI, 0)
    background_ROI.append(np.zeros((h, w), dtype = np.float32))
    
    # Find initial background for the ROI
    crop_width, crop_height = get_ROI_size(ROI, 0)
    bFrames = 20
    stepFrames = 360 # Check background frame every 3 seconds
    startFrame=f-(math.floor(stepFrames*math.floor((bFrames/2))))
    endFrame=f+(math.ceil(stepFrames*math.floor((bFrames/2))))
    
    # check we do not run off the end of the video; run the 30 second window back back if needed, 'stepFrames' at a time
    nF = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
    while(endFrame>nF):
        endFrame-=stepFrames
        startFrame-=stepFrames
        
    backgroundStack = np.zeros((crop_height, crop_width, bFrames), dtype = np.float32)  
    previous = np.zeros((crop_height, crop_width), dtype = np.float32)
    vid.set(cv2.CAP_PROP_POS_FRAMES,f) 
    ret, im = vid.read()
    current = np.float32(cv2.cvtColor(im, cv2.COLOR_BGR2GRAY))
    crop, xOff, yOff = get_ROI_crop(current, ROI,0)
    backgroundStack[:,:,0] = np.copy(crop)
    previous = np.copy(crop)
    bCount = 0
    changes=[]
    for i in range(startFrame,endFrame,stepFrames):
        vid.set(cv2.CAP_PROP_POS_FRAMES,i) 
        ret, im = vid.read()
        current = np.float32(cv2.cvtColor(im, cv2.COLOR_BGR2GRAY))
        crop, xOff, yOff = get_ROI_crop(current, ROI, 0)
    
        # Measure change from current to previous frame
        absdiff = np.abs(previous-crop)
        level = np.median(crop)/7
        change = np.mean(absdiff > level)
        changes.append(change)
        previous = np.copy(crop)
        # If significant, add to stack...possible finish
        if(change > 0): # currently set to 0 (i.e. doesn't care if significant change)
            backgroundStack[:,:,bCount] = np.copy(crop)
            bCount = bCount + 1
            #print(bCount)
            if(bCount == bFrames):
                break
    
    # Compute background
    backgroundStack = backgroundStack[:,:, 0:bCount]
    BG = np.uint8(np.median(backgroundStack, axis=2))
    
    # Return updated background
    return BG,changes
        
        
def compute_initial_background(vid, ROI):

    numFrames = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))-100 # Skip, possibly corrupt, last 100 frames (1 second)
    # Allocate space for ROI background
    background_ROI = []
    w, h = get_ROI_size(ROI, 0)
    background_ROI.append(np.zeros((h, w), dtype = np.float32))
    
    # Find initial background for the ROI
    crop_width, crop_height = get_ROI_size(ROI, 0)
    bFrames = 20
    #stepFrames = np.int(np.floor_divide(np.floor(numFrames*0.05),bFrames)) # Check background frame uniformly across time series for [numSteps] frames
    stepFrames = 360 # Check background frame every 3 seconds
    
    backgroundStack = np.zeros((crop_height, crop_width, bFrames), dtype = np.float32)  
    previous = np.zeros((crop_height, crop_width), dtype = np.float32)

    # Store first frame
    vid.set(cv2.CAP_PROP_POS_FRAMES, 0) # skip first 200 frames
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
        level = np.median(crop[crop>30])/6
        change = np.mean(absdiff > level)
        changes.append(change)
        previous = np.copy(crop)
        #print(change)
        # If significant, add to stack...possible finish
        if(change > 0): # currently set to 0 (i.e. doesn't care if significant change)
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
def get_largest_contour(contours,mask,area_limit=240,length_limit=100):
    # Find contour with maximum area and store it as best_cnt
    max_area = 0
    for cnt in contours:
        area = cv2.contourArea(cnt)
        mask=np.zeros(mask.shape)
        cv2.drawContours(mask,[cnt],0,1,-1) # -1 draw the contour filled
        pixels = np.nonzero(mask)
        Ylength=max(pixels[0])-min(pixels[0])
        Xlength=max(pixels[1])-min(pixels[1])
            
        # check the contour isn't a silly shape (and therefore unlikely to be a fish)
        if area > max_area and area<area_limit and Ylength<length_limit and Xlength<length_limit:
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