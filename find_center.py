
#Source: https://www.codementor.io/innat_2k14/extract-a-particular-object-from-images-using-opencv-in-python-jfogyig5u

# Capture the mouse click events in Python and OpenCV
'''
-> draw shape on any image 
-> reset shape on selection
-> crop the selection
run the code : python capture_events.py --image image_example.jpg
'''


# import the necessary packages
import cv2
from os import listdir
import sys


CROP_SIZE = 160
classes = dict([('pinky',0), ('elle',1), ('yo',2), ('index',3), ('thumb',4)])


def shape_selection(event, x, y, flags, param):
    # grab references to the global variables
    global ref_point, image, clone, placing_rectangle, done_rectangle

    if event == cv2.EVENT_LBUTTONDOWN:
        image = clone.copy()
        placing_rectangle = True
    
    elif event == cv2.EVENT_MOUSEMOVE:
        if placing_rectangle:
            image = clone.copy()
            cv2.rectangle(image, (x,y), (x-CROP_SIZE, y-CROP_SIZE), (0, 255, 0), 2)
            cv2.imshow("image", image)
    
    elif event == cv2.EVENT_LBUTTONUP:
        placing_rectangle = False
        done_rectangle = True
        cv2.rectangle(image, (x,y), (x-CROP_SIZE, y-CROP_SIZE), (0, 255, 0), 2)
        ref_point = [(x-CROP_SIZE, y-CROP_SIZE), (x,y)]
        cv2.imshow("image", image)
        

ref_point = []
names = [name for name in listdir("./dump/img_src/") if "tiff" in name]
refpoints_file = open("./dump/refpoints.txt","a") 
refpoints_file.write("\n===New encoding session===\n") 
quit_program = False

for name in names:
    placing_rectangle = False
    done_rectangle = False
    
    # load the image, clone it, and setup the mouse callback function
    image = cv2.imread("./dump/img_src/" + name)
    clone = image.copy()
    cv2.namedWindow("image")
    cv2.setMouseCallback("image", shape_selection)
    
    # keep looping until the 'q' key is pressed
    while True:
        # display the image and wait for a keypress
        cv2.imshow("image", image)
        key = cv2.waitKey(1) & 0xFF
    
        if done_rectangle:
            break
      
        if key == ord("e"):
            quit_program = True
            break
    
    crop_img = clone[ref_point[0][1]:ref_point[1][1], ref_point[0][0]:ref_point[1][0]]
    crop_img = cv2.resize(crop_img, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_CUBIC)
    cv2.imwrite('./dump/img_cropped/'+name, crop_img)
    
    refpoints_file.write(str(ref_point[1])+"\n") 
    
    # close all open windows
    cv2.destroyAllWindows()
    
    if quit_program:
        refpoints_file.close()
        sys.exit()
    
    
refpoints_file.close()