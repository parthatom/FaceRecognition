import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from align import AlignDlib

#get_ipython().magic(u'matplotlib inline')

def load_image(path):
    img = cv2.imread(path, 1)
    # OpenCV loads images with color channels
    # in BGR order. So we need to reverse them
    return img[...,::-1]

# Initialize the OpenFace face alignment utility
alignment = AlignDlib('models/landmarks.dat')

# Load an image of Jacques Chirac
jc_orig = load_image("./images/ABDUL_AHAD/ABDUL_AHAD0001.jpg")
scale_percent = 60 # percent of original size
width = int(jc_orig.shape[1] * scale_percent / 100)
height = int(jc_orig.shape[0] * scale_percent / 100)
dim = (width, height)
# resize image
jc_orig = cv2.resize(jc_orig, dim, interpolation = cv2.INTER_AREA)
bb = alignment.getLargestFaceBoundingBox(jc_orig)
jc_aligned = alignment.align(96, jc_orig, bb, landmarkIndices=AlignDlib.OUTER_EYES_AND_NOSE)
cap = cv2.VideoCapture(-1)
loop = 5e10
while(True):
    # Capture frame-by-frame
    ret, jc_orig = cap.read()
    print(jc_orig.shape)
    #jc_orig = cv2.resize(jc_orig, dim, interpolation = cv2.INTER_AREA)
    bb = alignment.getLargestFaceBoundingBox(jc_orig)
    print(bb)
    if(bb==None):
        continue
		

    jc_aligned = alignment.align(96, jc_orig, bb, landmarkIndices=AlignDlib.OUTER_EYES_AND_NOSE)
    # Our operations on the frame come here
    
	
    # Display the resulting frame
    cv2.imshow('frame',jc_aligned)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
# Detect face and return bounding box

# Transform image using specified face landmark indices and crop image to 96x96


# Show original image
"""plt.subplot(131)
plt.imshow(jc_orig)

# Show original image with bounding box
plt.subplot(132)
plt.imshow(jc_orig)
plt.gca().add_patch(patches.Rectangle((bb.left(), bb.top()), bb.width(), bb.height(), fill=False, color='red'))

# Show aligned image
plt.subplot(133)
plt.imshow(jc_aligned"""

"""print(jc_aligned.shape)
cv2.imshow(" ",jc_aligned)
cv2.imshow("orig", jc_orig)
cv2.waitKey(0)"""
