DATAROOT = r'D:\\Movies\DataForAdam\DataForAdam\GroupedTracking'
LIBROOT = r'C:\\Users\thoma\OneDrive\Documents\GitHub\Arena_Zebrafish'
MOVIEROOT = r'S:\\WIBR_Dreosti_Lab\Tom\RawMovies'

# Set library paths
import sys
lib_path = LIBROOT + "\ARK\libs"
ARK_lib_path = LIBROOT + "\libs"
sys.path.append(lib_path)
sys.path.append(ARK_lib_path)

# Import useful libraries
import glob
import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from sklearn.decomposition import PCA
import cv2
import ARK_utilities
import ARK_bouts

# Reload libraries
import importlib
importlib.reload(ARK_utilities)
importlib.reload(ARK_bouts)

# Helper functions
# Extract crop
def extract_crop(video, index, X, Y, crop_size):

    width=int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height=int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Extract bout start and stop frames
    video.set(cv2.CAP_PROP_POS_FRAMES, index)
    ret, frame = video.read()
    grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Add border
    border = np.zeros((height+crop_size*2, width+crop_size* 2))
    border[crop_size:(crop_size+width), crop_size:(crop_size+height)] = grayscale

    # Crop around fish
    c1 = int(X) 
    r1 = int(Y)
    c2 = int(X + (crop_size*2)) 
    r2 = int(Y + (crop_size*2))
    crop = border[r1:r2, c1:c2]

    return crop

# Animation function
def play_clip(i, video, start_index, X, Y, crop_size):
    crop = extract_crop(video, start_index+i, X[start_index],Y[start_index],crop_size)
    plt.imshow(crop, cmap='gray')

# -------------------------------------
# Get tracking file
#tracking_path = '/home/kampff/Data/Arena/ExemplarFish/Lesion/Tracking/200319_EmxGFP_Asp_M0_3_tracking.npz'

# Get movie file
#movie_path = '/home/kampff/Data/Arena/ExemplarFish/Lesion/Movies/200319_EmxGFP_Asp_M0_3.avi'


# Make list of tracking files and associate movies from the server
    
Trackings_1 = glob.glob(DATAROOT + r"\EC_M0\*tracking.npz")
Trackings_1.remove(Trackings_1[8])
Trackings_1.remove(Trackings_1[18])
Movies_1 = []
for tracking in Trackings_1:
    dirr,name=tracking.rsplit(sep='\\',maxsplit=1)
    pieces=name.split(sep='_')
    date=pieces[0]
    name=name[0:-13]
    Movies_1.append(glob.glob(MOVIEROOT + r"\\" + date + r'\\' + name + '*.avi'))
# Pick a tracking file
n=5
movie_path=Movies_1[n]
movie_path=movie_path[0]
tracking_path=Trackings_1[n]

# Parameters
FPS=120

# Bout PCA
bout_trajectories = []

# Load a tracking example (fx, fy, bx, by, ex, ey, area, orientation, speed)
tracking = np.load(tracking_path)['tracking']
num_frames = tracking.shape[0]

# Extract tracking
X = tracking[:,2]
Y = tracking[:,3]
A = tracking[:,7]

# dX, dY, dA(ngle)
dX = np.diff(X, prepend=X[0])
dY = np.diff(Y, prepend=Y[0])
dA = ARK_utilities.diffAngle(A)
dA = ARK_utilities.filterTrackingFlips(dA) / (2*np.pi)

# Analyze bouts
bouts = ARK_bouts.analyze(tracking)
bouts = ARK_bouts.filterTinyBouts(bouts)
num_bouts = bouts.shape[0]

# Plot bout scatter
plt.plot(bouts[:, 4], bouts[:, 5], '.')
plt.show()

# Load tracking movie
vid = cv2.VideoCapture(movie_path)
width=int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
height=int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
numFrames = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))

# Display individual bouts and classifications
rand=np.random.randint(0,500)
for i in range(rand,rand+100): #np.random.permutation(np.arange(num_bouts)):

    bout = bouts[i]

    # Extract bout features
    start = int(bout[0])
    peak = int(bout[1])
    stop = int(bout[2])
    duration = int(bout[3])
    net_angle = bout[4]
    net_distance = bout[5]

    # Extract crop around fish
    crop_window_size = 50
    crop_start = extract_crop(vid, start, X[start], Y[start], crop_window_size)
    crop_stop = extract_crop(vid, stop, X[stop], Y[stop], crop_window_size)

    # Prepare bout trajectory
    tX_start = X[start:stop] - X[start] + crop_window_size
    tY_start = Y[start:stop] - Y[start] + crop_window_size
    pX_start = X[peak] - X[start] + crop_window_size
    pY_start = Y[peak] - Y[start] + crop_window_size
    tX_stop = X[start:stop] - X[stop] + crop_window_size
    tY_stop = Y[start:stop] - Y[stop] + crop_window_size
    pX_stop = X[peak] - X[stop] + crop_window_size
    pY_stop = Y[peak] - Y[stop] + crop_window_size

    fig = plt.figure()
    plt.subplot(1,3,1)
    plt.title("Angle: {0:.2f}".format((net_angle)))
    plt.imshow(crop_start, cmap='gray')
    plt.plot(tX_start, tY_start, 'b')
    plt.plot(tX_start, tY_start, 'y.')
    plt.plot(tX_start[0], tY_start[0], 'go')
    plt.plot(tX_start[-1], tY_start[-1], 'ro')
    plt.plot(pX_start, pY_start, 'm+')
    plt.subplot(1,3,2)
    plt.title("Distance: {0:.2f}".format((net_distance)))
    plt.imshow(crop_stop, cmap='gray')
    plt.plot(tX_stop, tY_stop, 'b')
    plt.plot(tX_stop, tY_stop, 'y.')
    plt.plot(tX_stop[0], tY_stop[0], 'go')
    plt.plot(tX_stop[-1], tY_stop[-1], 'ro')
    plt.plot(pX_stop, pY_stop, 'm+')
    plt.subplot(1,3,3)
    anim = FuncAnimation(fig, func=play_clip, fargs=(vid, start, X, Y, crop_window_size), frames=duration, repeat=False, interval=20, blit=False)
    plt.show()
    plt.waitforbuttonpress()
    plt.close()
#FIN
