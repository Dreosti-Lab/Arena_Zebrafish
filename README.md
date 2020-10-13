# Arena_Zebrafish
Collection of scripts and functions to analyse "Arena zebrafish" data

Pipeline:

Step1_ArenaTracking: 

	Performs batch tracking analysis on groups of data folders. Cycles through each movie, automatically creating a file structure containing tracking data, tracking figures and a cropped movie.
	Tracking is performed using opencv functions and custom written functions in the AZ_video library.

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

	Inputs: 
	
		1. The complete path to a folder list in plain text format: The first line must be the root directory of the data (avi files), followed by each subdirectory on each line (see example file). 
	
		Preprocessing with ROIs drawn in Bonsai prior to saving the movies are automatically detected if they exist within the data folder and utilised. 
		Only one ROI is compatible at this stage (use to crop edges)

	Outputs:

		This script generates 

			.npx file containing tracking data over the length of the movie is saved in a new Tracking folder in the corresponding data folders: 

						TrackingFile contains:			  heading
							    				  'fish' pixels area
											  motion energy (absolute difference between frame intensities above threshold)
						                                          x and y coordinates for 
												the eye centroid (darkest 10% of 'fish' pixels)
												the fish body centroid (darkest 50% of 'fish' pixels)
												the fish centroid (centroid of all 'fish' pixels)


			Figures saved in a new 'Figures' folder within corresponding data folders:

						Figures are:				   A supplementary movie of a small area (128,128 by default) around the fish								
											   initial and final backgrounds (there should be no fish shadow in these images)
											   background image overlaid with tracking coordinates for first frame and all frames.

	Options:	

	Named parameters of the tracking algorithm can be modified within this script. Named parameters and their default values are shown:

	plot=1			#	Value of 1 generates images to monitor tracking progress in real time. Disable (Value of 0) to speed up algorithm c.20%
	cropOp=1		#	Value of 1 uses a small cropped area (defined by cropSize) around the fish once it has been found on the first frames. This increases the speed of the algorithm approximately 5-fold compared to using the entire frame of a 1200x1200 movie.  Disable by setting to 0
	FPS=120			#	Frames per second of acquired movie
	saveCroppedMovie=1	#	Value of 1 generates the cropped movie and saves it in the Figures folder. Disable (Value of 0) if not needed. Only applies if cropOp == 1
	cropSize=[128,128]	#	Dimensions in pixels of cropped area (diameter) around fish to use for tracking and cropped movie generation 
	startFrame=0		#	which frame of the movie to consider the first frame. Can be used to remove artifacts or corrupted frames from the start of a movie.

	Data management:
		
		avi files that are NOT successfully tracked (>10% of frames where fish was lost) have a shortcut saved in a new FailedAvis folder in the data folder, or FinishedAviFiles if tracking WAS successful.
		
Step2_ArenaAnalysis:

	Performs bout detection and overall analysis on a fish by fish basis, Computing various metrics (listed below) and saving all data in a single 'dictionary' object in a new Analysis folder. Generates figures for each individual fish saved in new Figures folders within corresponding datafolders.
	Bouts are detected using the differentiated trace of a smoothed sum of instantaneous angular and radial velocities, weighted by SD and called motion_signal. A threshold is computed as 0 + 5*sigma of a Gaussian fitted to the lowest 90% of motion_signal values.  
	Can be used to group fish simultaneously if all analysed fish belong to the same group. For details of grouping fish see Step3.

	Inputs:

		Data:	EITHER 

				Complete path to a folderList file as in Step1

			OR

				Complete path to a folder containing shortcuts to trackingFiles generated in Step1. Correct aviFiles, analysis dictionaries etc are found using the target of these shortcuts. 

		Flags (names and default values):	
			createDict=True			#	Create dedicated analysis dictionary for this fish
			createFigures=True		#	Generate and save figures for this fish
			keepFigures=False		#	If True, Do not close figures after analysis (not recommended unless analysing a single fish)
			group=False			#	Simultaneously group fish together. If 'True' a groupname must be defined (see other inputs below)
			createGroupFigures=True		#	Only applied if group==True. Creates figures of grouped analysis (described in Step3)
			keepGroupFigures=False		#	Do not close group figures after analysis

		Other inputs (names and default values):
			FPS = 120			#	Frames per second of acquired movie
			groupName=''			#	Only applied if group==True. User defined name (string) for grouped fish.
			
	Outputs:
		Dictionaries generated are saved in a new Analysis folder in corresponding dataFolders and are structured as follows:

			SingleFish  =       {'info' :   {'Date'                 :   date,				# date of experiment
        	                             		 'Genotype'             :   gType,				# genotype of fish
                	                     		 'Chamber'              :   chamber,				# chamber used in experiment
                        	             		 'Condition'            :   cond,				# experimental condition
                                	     		 'FishNo'               :   fishNo,				# experiment number
                                     			 'TrackingPath'         :   trackingFile,			# path of trackingFile
                                     			 'AviPath'              :   aviFile				# path of aviFile
	                                     		 },
        	                 	     'data' :   {'BPS'                  :   BPS,				# Average Bouts Per Second
                	                     		 'avgVelocity'          :   avgVelocity,			# Average velocity (mm/s)
                        	             		 'avgAngVelocityBout'   :   avgAngVelocityBout,			# Average angular velocity (degrees/s) (computed by bouts, not continuous measurement)
                                	     		 'biasLeftBout'         :   bias,				# Angular velocity magnitude bias (ratio of bout turn magnitudes). Positive values indicate a left bias 
                                     			 'LTurnPC'              :   LturnPC,				# Percentage of turns (not including forward swims) that are left turns
                                     			 'distPerFrame'         :   distPerFrame,			# Distance travelled per frame (mm) 
	                                     		 'cumDist'              :   cumDist,				# Cumulative distance travelled per frame (mm)
        	                             		 'heatmap'              :   heatmap,				# 2D histogram (10 x 10) of time spent.
                	                     		 'avgBout'              :   {'Mean'             :   avgBout,	# Average trace of all detected bouts
                        	             		                             'SD'               :   avgBoutSD	# SD of average trace of all detected bouts
                                	    		                              },
	                                    		 'boutAmps'             :   boutAmps,				# List of maximum velocities for all detected bouts
        	                            		 'boutDists'            :   boutDists,				# List of total distances travelled for all detected bouts
                	                    		 'boutAngles'           :   boutAngles,				# List of net heading changes for all detected bouts. Positive values indicate left turns
                        	             		 'boutSeq'              :   {'Lturns'   :   LTurns,		# Boolean arrays indicating whether a bout was a turn (above 10 degrees) to the left...
                                	           	                             'RTurns'   :   RTurns,		# ... to the right...
                                        	                         	     'FSwims'   :   FSwims		# ... or was a forward swim (< 10 degrees) 
                                                	                 	    },
	                                     		 'allBouts'             :   allBoutsList,			# List of traces of motion energy for all detected bouts
        	                             		 'allBoutsDist'         :   allBoutsDistList,			# List of traces of distance per frame for all detected bouts
                	                     		 'allBoutsOrts'         :   allBoutsOrtList,			# List of traces of heading per frame for all detected bouts. Headings are rotated so that start position = 0
                        	             		}
                        		   }

			Utilising the 'info' functionality of the dictionary structure is optional and these fields are empty by default. To use it, experimental files should be named in the following way:
				date_gType_cond_chamber_fishNo
	
					date should be 6 digits in reverse order (yymmdd)
					gType is any string defining this fish's genotype
					cond is any string defining the experimental condition of the fish
					chamber is a user defined code for the chamber used for this experiment
					fishNo is any string defining the trial/repeat/fish identifier

				e.g. movie name utilising the info functionality:

				200220_EmxGFP_Asp_M0_1.avi


		Figures generated for each fish are saved in a new Analysis\Figures folder in corresponding dataFolders:
			avgBout		# 	average trace of all detected bouts with SD
			boutAmps	#	histograms of bout amplitudes
			heatmaps	#	heatmaps of time spent in 10 x 10 grid of movie.
			cumDist		# 	cumulative distance travelled
			boutFinder	#	trace of distance per frame for the first 90 seconds of the movie. Red lines mark where bouts were detected, shaded areas are all regions where animal is considered mid-bout.

		Data Management:
			If the 'info' functionality is used, a new folder structure is created at a user defined path ('root' named parameter in AZU.createShortcutTele, near end of Step2 script) where shortcuts to tracking files of 
			fish with the same genotypes, conditions and chambers are created. This enables more automated grouping of fish at step3. By default the root of this structure will be r'D:\\Movies\\Processed\\'
			
		
Step2.1_spatialWindows:

	Script to segregate movies into arbitrary, user defined, polygonal ROIs (currently only supports 9 ROIs). The first frame of each movie will appear to the user. ROIs can be drawn using the left mouse button to define an anchor point,
	then adding anchors unti the polygon is finished. Right clicking will generate and save the ROI; user then defines a new ROI. ROIs can have as many anchor points as needed, and ROIs do not have to have the same number of anchor points. 
	If 'info' functionality has been used on single fish, user can define a path to a previously generated ROIs saved in .npy format according to the 'chamber' used. This functionality will be expanded as more chambers are used.

	Inputs:
		Data:
			As for step2, a path to a folderListFile OR the path to a folder containing shortcuts to tracking files (e.g. those automatically generated and organised into groups in step2). 
	
		Flags (names and default values):
			createSpatialFigures=True	# Generate and save figures for each ROI
			keepSpatialFigures=True		# If True, Do not close figures after analysis (not recommended unless analysing a single fish)
			sameROIs = False		# Use the same ROIs for all fish in this group. If False, user is able to intervene in the analysis loop each new file, choosing to use previous ROIs or define new ones for each movie
		
		Other inputs (names and default values):
			ROINames=['Top Left','Top','Top Right','Middle Left','Central Chamber', 'Middle Right', 'Bottom Left','Bottom','Bottom Right'] # complete list of names for each ROI (currently only supports 9 ROIs)
	

	Outputs: 
		ROI data is added to the dictionary file created for individual fish and saved with a filename appended with '_ROIs'. A new key is generated within the single fish dictionary ['ROIs']. SingleFish['ROIs'] is a list of dictionaries,
		one for each ROI, and is structured as follows (Variables described previously not described here)

			ROIdata	=	{'ROIName'              :   ROIName,					# String containing name of this ROI (defined in ROINames in Inputs)
		                         'BPS'                  :   BPS,					
                		         'avgVelocity'          :   avgVelocity,				
		                         'avgAngVelocityBout'   :   avgAngVelocityBout,				
		                         'biasLeftBout'         :   biasLeftBoutS,				
		                         'LTurnPC'              :   LTurnPCS,					
		                         'distPerFrame'         :   distPerFrame,				
		                         'cumDist'              :   cumDistS[r],				
		                         'avgBout'              :   {'Mean'             :   avgBoutS[r],	
		                                                     'SD'               :   avgBoutSDS[r],	
		                                                     },
		                         'boutAmps'             :   boutAmps,
		                         'boutDists'            :   boutDist,
		                         'boutAngles'           :   boutAngles,
		                         'allBouts'             :   allBouts,
		                         'boutOrts'             :   allBoutsOrt,
		                         'ROIMask'              :   thisMask,					# Binary mask defining this ROI
		                         'PCTimeSpent'          :   timeInROI_PC[r],            		# Percentage time spent in this ROI
		                         'BoutSeq'              :   {'Left'     :   LTurnS[r],			
		                                                     'Right'    :   RTurnS[r],
		                                                     'Forward'  :   FSwimS[r]}})
		