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
import scipy.ndimage
import math
import cv2
import AZ_utilities as AZU

# Process Video : Track fish in AVI
def arena_fish_tracking(aviFile, output_folder, ROI,plot=True,cropOp=1,FPS=120,saveCroppedMovie=True,startFrame=0,display=False,trackTail=True,larvae=True):
    if larvae:
        gauss_filt=3
        tailThreshold=10
        kernelSize=5
        areaLimit=285
        cropSize=[128,128]
        BGstartFrame=(1*60)*FPS # frame of movie with which to start computing the background... avoids having fish in background if frozen for first few minutes after transfer
    else:
        gauss_filt=5
        tailThreshold=15
        kernel=12
        areaLimit=2000
        cropSize=[256,256]
        BGstartFrame=(3*60)*FPS # frame of movie with which to start computing the background... avoids having fish in background if frozen for first few minutes after transfer
    #    l=0
    arcRad=8 # this is the initial radius of the arcs that will find the tip of the tail
    arcNum=7 # this is the total number of segments the algorithm will divide the tail into once it's found the tip
    arcSam=2.5 # this is the distance between angle samples the draw the circle and arcs that find the body and tail
    
    if plot:
        pmsg='ON'
    else:
        pmsg='OFF'
    if cropOp==1:
        cmsg='ON'
    else:
        cmsg='OFF. WARNING!! Tracking can take a very long time with cropping turned OFF'
    if saveCroppedMovie:
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
    
    backgroundFull = compute_initial_background(vid, ROI,startFrame=BGstartFrame)
    
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
    maxNumFrames=432000#144000#(5*60)*120#144000#120*60*5
    numFrames = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))-100 # Skip, possibly corrupt, last 100 frames (1 second)
    numFrames = numFrames - startFrame
    if numFrames>maxNumFrames:numFrames=maxNumFrames
    previous_ROI=[]
    previous_ROI.append(np.zeros((h,w),dtype = np.uint8))

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
    fishSegX_allFramesT=[]
    fishSegY_allFramesT=[]
    for f in range(numFrames):
#        print(f)
        # Report Progress every 120 frames of movie
#        if (f%120) == 0:
#            print ('\r'+ str(f) + ' of ' + str(numFrames) + ' frames done')
        
        if f==0 and saveCroppedMovie:
            print('Creating cropped Movie')
            croppedSegPath=output_folder + '\\' + expName + '_tail_segmented_croppedInt.avi'
            croppedTailSegMovOut = cv2.VideoWriter(croppedSegPath, cv2.VideoWriter_fourcc(*'DIVX'), FPS, (cropSize[1],cropSize[0]))
            croppedSegPath=output_folder + '\\' + expName + '_croppedInt.avi'
            croppedMovOut = cv2.VideoWriter(croppedSegPath, cv2.VideoWriter_fourcc(*'DIVX'), FPS, (cropSize[1],cropSize[0]))
            
        # Read next frame 
        ret, im = vid.read()
        errF=-1
    
        # Convert to grayscale (uint8)
        current = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        crop, xOff, yOff = get_ROI_crop(current, ROI,0)
        background, xOff, yOff = get_ROI_crop(backgroundFull, ROI,0)                      
        diffimg= np.subtract(background,crop)                
        diffimg[diffimg>220] = 0  #remove nearly saturated differences; at edges of objects (not sure where these come from... water evaporation maybe?)
            
        # Determine current threshold
        # Because the edges are much lower in intensity, this can mess with the threshold if too much of the image is black
        # therefore we only include values above 70 (lowest measured pixel value of navigable chamber at set IR intensity)
        threshold_level = np.median(background[background>70])/6   
        tailThreshold = threshold_level*0.7
        # Threshold            
        level, threshold = cv2.threshold(diffimg,threshold_level,255,cv2.THRESH_BINARY)
        threshold = np.uint8(diffimg > threshold_level)
         
        # Binary Close
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(kernelSize,kernelSize))
        closing = cv2.morphologyEx(threshold, cv2.MORPH_CLOSE, kernel)
            
        # Find Binary Contours            
        contours, hierarchy = cv2.findContours(closing,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
            
        # Create Binary Mask Image
        mask = np.zeros(crop.shape,np.uint8)
        
        # If there are NO contours, then skip frame
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
            largest_cnt, area = get_largest_contour(contours,area_limit=areaLimit)
            
            # If the particle to too small to consider, skip frame
            if area == 0.0 or area<0:
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
                    if trackTail:
                        for j in range(arcNum):
                            nextX=fishSegX_allFramesT[-1]
                            nextY=fishSegY_allFramesT[-1]   
                            fishSegX_allFramesT.append(nextX)
                            fishSegY_allFramesT.append(nextY)
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
                    if trackTail:
                        for j in range(arcNum):
                            fishSegX_allFramesT.append(-1)
                            fishSegY_allFramesT.append(-1)
            else:
                # Draw contours into Mask Image (1 for Fish, 0 for Background
                cv2.drawContours(mask,[largest_cnt],0,1,-1) # -1 draw the contour filled
                pixels = np.nonzero(mask)
                
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
                area = float(area)
                
                # Highlight 80% of the birghtest pixels (body + eyes)                    
                numBodyPixels = int(np.ceil(area*0.8))
                    
                # Highlight 10% of the birghtest pixels (mostly eyes)     
                numEyePixels = int(np.ceil(area*0.1))
                    
                # Fish Pixel Values (difference from background)
                fishValues = diffimg[pixelpoints[:,0], pixelpoints[:,1]]
                sortedFishValues = np.sort(fishValues)
                
                bodyThreshold = sortedFishValues[-numBodyPixels]                    
                eyeThreshold = sortedFishValues[-numEyePixels]
#                tailThreshold = sortedFishValues[0]/2
                
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
                fX = float(np.sum(c*values))/acc
                fY = float(np.sum(r*values))/acc
                
                # Eye Centroid (a weighted centorid)
                values = np.copy(all_values)                   
                values = (values-eyeThreshold+1)
                values[values < 0] = 0
                acc = np.sum(values)
                eX = float(np.sum(c*values))/acc
                eY = float(np.sum(r*values))/acc
                
                # Body Centroid (a binary centroid, excluding "eye" pixels)
                values = np.copy(all_values)                   
                values[values < bodyThreshold] = 0
                values[values >= bodyThreshold] = 1                                                            
                values[values > eyeThreshold] = 0                                                            
                acc = np.sum(values)
                bX = float(np.sum(c*values))/acc
                bY = float(np.sum(r*values))/acc
#                sampleAngle=arcSam
#                circleCoords=AZU.findCircleEdgeSubPix(xo=eY,yo=eX,r=18,sampleN=int(360/sampleAngle))
#                circleValsY=[]
#                for idx,a in enumerate(circleCoords):
##                        circleValsX.append(idx*sampleAngle)
#                    if a[0]>=h: a[0]=h-1
#                    if a[1]>=w: a[1]=w-1
#                    circleValsY.append(getSubPixelIntensity(a,diffimg))
#                bX=circleCoords[np.argmax(circleValsY)][1]
#                bY=circleCoords[np.argmax(circleValsY)][0]
                    
                    # Now use this as the seed to find body contour (might be a better way to do this in cv2)
#                    reg=findBodyFromSeed(60,diff,seed)

# ---------------------------------------------------------------------------------         
                # Heading (0 deg to right, 90 deg up)
                heading = math.atan2((bY-eY), (eX-bX)) * (360.0/(2*np.pi))

# -------------Tail Tracking ---------------------------------------------------------------------------
                if trackTail:
                    # Find tail in segments starting at the body and using the heading
                    if f==0:
                        arcHeading=heading
                        prevsX=bX
                        prevsY=bY
                        tailLength=0
                        arcRadt=arcRad
                        print('finding end of tail...')
                        flag=True
                        while flag and f==0: # find length of tail if on first frame
                            arcVals,arcX,arcY=AZU.findArc(prevsY,prevsX,arcHeading,diffimg,arcLen=120,arcRad=arcRadt*0.5,arcSam=arcSam)
                            [h,w]=diffimg.shape
                            arcVals_sm=scipy.ndimage.gaussian_filter1d(arcVals,gauss_filt)
                            # could then fit a gaussian but probably unneccessary
                            while np.max(arcVals_sm)<tailThreshold and arcRadt>=0:
                                # if we lose the tail, shrink the segment until you find the tip (or the segment gets too small)
                                arcRadt-=1
                                arcVals,arcX,arcY=AZU.findArc(prevsY,prevsX,arcHeading,diffimg,arcRad=arcRadt)
                                arcVals_sm=scipy.ndimage.gaussian_filter1d(arcVals,5)
                                flag=False
                            tailLength+=arcRadt
                            
                            # find new point (subPixel peak in arc profile)
                            nextSegsX=arcX[np.argmax(arcVals_sm)]+xOff
                            nextSegsY=arcY[np.argmax(arcVals_sm)]+yOff
                      
                            # update angle between the new point and the previous point to feed back into the loop (note it is reversed as the heading between body and eye is reversed in findArc)
                            arcHeading=math.atan2((nextSegsY-prevsY), (prevsX-nextSegsX)) * (360.0/(2*np.pi))
                            prevsX=nextSegsX
                            prevsY=nextSegsY
                        # Once tail length has been found, divide this by the desired number of segments to get optimum arc radius (should be close to 10 pixels)
                        arcRad=int(np.floor(np.divide(tailLength,arcNum)))
                        print('found end of tail...tailLength=' + str(tailLength))
                    # print('out of while loop')
                    # repeat arcs to find segment coordinates with optimised radius 
                    # make copy of current, add markers for segment coordinates (and arcs)
                    prevX=bX
                    prevY=bY
                    arcHeading=heading
                    currentMarkers=np.copy(current)
                    currentMarkers = cv2.cvtColor(currentMarkers, cv2.COLOR_GRAY2BGR)
                    currentMarkers=cv2.drawMarker(currentMarkers, (int(bX+xOff), int(bY+yOff)), (0,255,255), cv2.MARKER_CROSS, 7, thickness=1, line_type=8)
                    if display:
                        cv2.namedWindow("Display",cv2.WINDOW_AUTOSIZE )
                    
                    # print('finding segments...')
                    for j in range(arcNum):
                        arcVals,arcX,arcY=AZU.findArc(prevY,prevX,arcHeading,diffimg,arcRad=arcRad)
                        arcVals_sm=scipy.ndimage.gaussian_filter1d(arcVals,gauss_filt)
                        nextX=(arcX[np.argmax(arcVals_sm)])
                        nextY=(arcY[np.argmax(arcVals_sm)])
                        arcHeading=math.atan2((nextY-prevY), (prevX-nextX)) * (360.0/(2*np.pi))                     
                        fishSegX_allFramesT.append(nextX + xOff)
                        fishSegY_allFramesT.append(nextY + yOff)
    
                        cv2.drawMarker(currentMarkers, (int(nextX+xOff), int(nextY+yOff)), (0,0,255), cv2.MARKER_DIAMOND, 3, thickness=1, line_type=8)
                        for i in range(len(arcX)):
                            currentMarkers=cv2.drawMarker(currentMarkers, (int(arcX[i]+xOff), int(arcY[i]+yOff)), (0,255,0), cv2.MARKER_SQUARE, 1, thickness=1, line_type=8)
                        prevX=nextX
                        prevY=nextY
                    
                
                # resize so it doesn't take up whole screen
                _,startIdX,startIdY,endIdX,endIdY=cropImFromTracking(vid,current,fX + xOff,fY + yOff,cropSize)
                resFull=currentMarkers[startIdY:endIdY,startIdX:endIdX]
                resFullNoMarker=current[startIdY:endIdY,startIdX:endIdX]
#                if display:
#                    res=cv2.resize(resFull, (0, 0), fx=2, fy=2) 
#                    cv2.imshow("Display", res)
                if f>0 and saveCroppedMovie:
#                    tailSegMovOut.write(currentMarkers)
                    croppedTailSegMovOut.write(resFull)
                    croppedMovOut.write(resFullNoMarker)
                # Now crop the movie around the fish
                # crop movie here and recompute diffimg if working with the first frame
                # find start and end positions for cropping around the located fish
                if cropOp:
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
        backgroundInterval=int(FPS*(2*60))
       
        if((f%(backgroundInterval)==0) & (f!=0)):
            current_background=np.copy(backgroundFull)
            backgroundFull,_=recomputeBackground(vid,f+startFrame,OrigROI,FPS)
            
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
            
# ---------------------------------------------------------------------------------
        # Plot Fish in Movie with Tracking Overlay?
        if plot:
            if (f%600==0): # every 5 seconds
                plt.clf()
                enhanced = cv2.multiply(current, 1)
                color = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
                plt.imshow(color)
                plt.axis('image')
                plt.plot(fxS[:,0],fyS[:,0],'b.', markersize = 1)
                plt.plot(exS[:,0],eyS[:,0],'r.', markersize = 1)
                plt.plot(bxS[:,0],byS[:,0],'co', markersize = 1)

        else:  
            if (f == 0) or (f == numFrames-1): # only plot this in the first and last frame to save the file
                
                plt.clf()
                enhanced = cv2.multiply(current, 1)
                color = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
                plt.imshow(color)
                plt.axis('image')
                plt.plot(fxS[:,0],fyS[:,0],'b.', markersize = 1)
                plt.plot(exS[:,0],eyS[:,0],'r.', markersize = 1)
                plt.plot(bxS[:,0],byS[:,0],'co', markersize = 1)

# ---------------------------------------------------------------------------------
# Save Tracking Summary
        if(f == 0):
            path=output_folder + '\\' + expName + '_initial_tracking.png'
            plt.savefig(path, dpi=300)
            
            plt.figure('backgrounds')
            plt.imshow(backgroundFull)
            path=output_folder+'\\' + expName +'_initial_background.png'
            plt.savefig(path, dpi=300)
            plt.close('backgrounds')
       
        if(f == numFrames-1):
            path=output_folder+'\\' + expName +'_final_tracking.png'
            plt.savefig(path, dpi=300)
            plt.figure('backgrounds')
            plt.imshow(backgroundFull)
            path=output_folder+'\\' + expName +'_final_background.png'
            plt.savefig(path, dpi=300)
            plt.close('backgrounds')
            
        if(f==math.floor(numFrames/2)):
            print('Halfway done tracking')
         
        ##### END OF FRAME LOOP #####
        # Convert list into array and reshape it ('break it down')
    x_seg_data=np.asarray(fishSegX_allFramesT).reshape((numFrames,arcNum))
    y_seg_data=np.asarray(fishSegY_allFramesT).reshape((numFrames,arcNum))
            
# -------------------------------------------------------------------------
# Close Video Files
    if saveCroppedMovie:
        croppedMovOut.release()
        croppedTailSegMovOut.release()
    vid.release()
    print('Finished tracking')
   
    # Return tracking data
    return fxS, fyS, bxS, byS, exS, eyS, x_seg_data,y_seg_data,areaS, ortS, motS,failedAviFiles, errF

###############################################################################
def computeTailCurvatures(x_data,y_data,verbose=False):
    
    
    
     # Determine number of frames, segments and angles
    num_frames = np.shape(x_data)[0]
    num_segments = np.shape(x_data)[1]
    num_angles = num_segments - 1
      
    # Allocate space for measurements - creates empty arrays to store data
    cumulAngles = np.zeros(num_frames)
    curvatures = np.zeros(num_frames)
    motion = np.zeros(num_frames)
    finalTailAngle = np.zeros(num_frames)
    ## Measure tail motion, angles, and curvature ##
    # Take x and y values of first frame for each segment
    prev_xs = x_data[0, :]
    prev_ys = y_data[0, :]
    
    ########### Start of frame loop (f) #################
    for f in range(num_frames):
        if verbose: print(f)
        delta_thetas = np.zeros(num_angles) # Make an array of zeros with the same size as num angles (num seg-1)
        prev_theta = 0.0 # set first theta to zero
        
        ############### Start of segment loop (a) ################
        for a in range(num_angles):
            dx = x_data[f, a+1] - x_data[f, a] # dx between each segment for the same frame
            dy = y_data[f, a+1] - y_data[f, a] # dy between each segment for the same frame
            theta = np.arctan2(dx, dy) * 360.0 / (2.0*np.pi) # calc arctangent bt dx and dy, convert to deg
            delta_thetas[a] = theta - prev_theta
            prev_theta = theta # prev theta is set to current theta
            
        # final tail angle (between body and tip)
        dx = x_data[f, -1] - x_data[f, 0] # dx between each segment for the same frame
        dy = y_data[f, -1] - y_data[f, 0] # dy between each segment for the same frame
        finalTailAngle[f] = np.arctan2(dx, dy) * 360.0 / (2.0*np.pi) # calc arctangent bt dx and dy, convert to deg
        ############### End of angle loop (a) ################
        
        cumulAngles[f] = np.sum(delta_thetas) # sum all angles for this frame
        curvatures[f] = np.mean(np.abs(delta_thetas)) # mean of abs value of angles
        
        
        # Measure motion
        diff_xs = x_data[f,:] - prev_xs # difference between current x and prev x from each segment and each frame
        diff_ys = y_data[f,:] - prev_ys
        motion[f] = np.sum(np.sqrt(diff_xs*diff_xs + diff_ys*diff_ys)) # motion as sqrt of sq diff of x & y
        
        # Store previous tail
        prev_xs = x_data[f, :] # So that we don't always take the 1st frame of the movie to calculate motion
        prev_ys = y_data[f, :]
    ####### End of frame loop (f) ###############
    return finalTailAngle,cumulAngles,curvatures,motion

# Crop a single frame using tracking coordinates and return cropped frame and crop indices    
def cropImFromTracking(vid,im,fx,fy,cropSize):

    if(math.isinf(fx)):
        print('here fx is inf')
        
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
def makeCroppedMovieFromTracking(fxS,fyS,vid,saveName,FPS,cropSize,color=False):
    AZU.setFrame(vid,0)
    width=cropSize[0]
    height=cropSize[1]
    if color:
        out = cv2.VideoWriter(saveName,cv2.VideoWriter_fourcc(*'DIVX'), FPS, (width,height))
    else:
        out = cv2.VideoWriter(saveName,cv2.VideoWriter_fourcc(*'DIVX'), FPS, (width,height),0)
    print('Writing cropped video')
    for i in range(len(fxS)-1):
        startIdX = math.ceil(fxS[i]) - math.ceil(cropSize[0] / 2)
        endIdX = math.ceil(fxS[i]) + math.ceil(cropSize[0] / 2)
        startIdY = math.ceil(fyS[i]) - math.ceil(cropSize[1] / 2)
        endIdY = math.ceil(fyS[i]) + math.ceil(cropSize[1] / 2)
        
        # Check we don't fall off the edge of the image
        # if it does set start and end to min max of image
        wFull = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
        hFull = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        if(startIdY<0):
            endIdY=cropSize[1]                
            startIdY=0
        if(startIdX<0):
            endIdX=cropSize[0]
            startIdX=0
        if(endIdX>wFull):
            startIdX=wFull-cropSize[0]
            endIdX=wFull
        if(endIdY>hFull):
            startIdY=hFull-cropSize[1]
            endIdY=hFull
            
        ret, im = vid.read()
        if color==False:
            im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
            newIm=im[startIdY:endIdY,startIdX:endIdX]
        else:
            imB,imG,imR = cv2.split(im)
            newB=imB[startIdY:endIdY,startIdX:endIdX]
            newG=imG[startIdY:endIdY,startIdX:endIdX]
            newR=imR[startIdY:endIdY,startIdX:endIdX]
            newIm=cv2.merge((newB,newG,newR))
        out.write(newIm)
    out.release()
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
    width = int(ROIs[numROi, 2])
    height = int(ROIs[numROi, 3])
    
    return width, height

def recomputeBackground(vid,f,ROI,FPS):
    
    print('Updating background (' + str((f/FPS)/60) + ' mins done)')
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
    
    # check we do not run off the end of the video; run the 30 second window back if needed, 'stepFrames' at a time
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
        try:
            current = np.float32(cv2.cvtColor(im, cv2.COLOR_BGR2GRAY))
        except:
            print('here')
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
        
        
def compute_initial_background(vid, ROI,startFrame=0):

    numFrames = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))-100 # Skip, possibly corrupt, last 100 frames (1 second)
    # Allocate space for ROI background
    background_ROI = []
    w, h = get_ROI_size(ROI, 0)
    background_ROI.append(np.zeros((h, w), dtype = np.float32))
    
    # Find initial background for the ROI
    crop_width, crop_height = get_ROI_size(ROI, 0)
    bFrames = 100
    #stepFrames = int(np.floor_divide(np.floor(numFrames*0.05),bFrames)) # Check background frame uniformly across time series for [numSteps] frames
    stepFrames = 2400 # Check background frame every 20 seconds
    
    backgroundStack = np.zeros((crop_height, crop_width, bFrames), dtype = np.float32)  
    previous = np.zeros((crop_height, crop_width), dtype = np.float32)

    # Store first frame
    vid.set(cv2.CAP_PROP_POS_FRAMES, startFrame) # skip first 200 frames
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
    r1 = int(ROIs[numROi, 1])
    r2 = int(r1+ROIs[numROi, 3])
    c1 = int(ROIs[numROi, 0])
    c2 = int(c1+ROIs[numROi, 2])
    crop = image[r1:r2, c1:c2]
    
    return crop, c1, r1

# Return largest (area) cotour from contour list
def get_largest_contour(contours,area_limit=240):
    # Find contour with maximum area and store it as best_cnt
    max_area = 0
    for cnt in contours:
        area = cv2.contourArea(cnt)
#        mask=np.zeros(mask.shape)
#        cv2.drawContours(mask,[cnt],0,1,-1) # -1 draw the contour filled
#        pixels = np.nonzero(mask)
#        Ylength=max(pixels[0])-min(pixels[0])
#        Xlength=max(pixels[1])-min(pixels[1])
            
        if area > max_area and area<area_limit:
            max_area = area
            largest_cnt = cnt

    if max_area > 0:
        return largest_cnt, max_area
    else:
        return -1,-1

# Compare Old (scaled) and New (non-scaled) background images
def compare_backgrounds(folder):

    # Load -Initial- Background Frame (histogram from first 50 seconds)
    backgroundFile = folder + r'/background_old.png'
    background_old = misc.imread(backgroundFile, False)
    
    backgroundFile = folder + r'/background.png'
    background = misc.imread(backgroundFile, False)
    absDiff = cv2.absdiff(background_old, background)

    return np.mean(absDiff)

def getSubPixelIntensity(a,image): # only works one pixel radius... consider convolutional method for variable kernel sizes
    
    [x,y]=a
    # round coordinates to find root pixel
    xR=int(np.round(x))
    yR=int(np.round(y))
    
    # mod to 1 (how much closer to the adjacent pixel am I?)
    remX=np.mod(x,1)
    remY=np.mod(y,1)
    
    # find how much closer you are to the diagonal pixel
    remD=np.sqrt(((1-remX)**2)+((1-remY)**2))
    
    # decide which direction we go to find adjacent pixels
    left=False
    down=False
    if x-xR<0:
        left=True
    if y-yR<0:
        down=True
    
    # find coordinates for adjacent pixels (y origin is top)
    if left:
        adjX=xR-1
    else:
        adjX=xR+1
    if down:
        adjY=yR+1
    else:
        adjY=yR-1
    
    # find intensity of root pixel
    iR=image[xR,yR]

    # weight root and add weighted points in each direction...
    #debug
#    print('y='+str(y))
#    print('x='+str(x))
#    print('adjX='+str(adjX))
#    print('adjY='+str(adjY))
#    print('remX='+str(remX))
#    print('remY='+str(remY))
#    
#    print('rootX='+str(xR))
#    print('rootY='+str(yR))
    
    valX=(iR*remX)+(image[adjX,yR]*(1-remX))
    valY=(iR*remY)+(image[xR,adjY]*(1-remY))
    valD=(iR*remD)+(image[adjX,adjY]*(1-remD))
    
    # return the mean
    return np.mean([valX,valY,valD])


# FIN