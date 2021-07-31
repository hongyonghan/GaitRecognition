import cv2
import numpy as np
import os

fpsVideo=30   #原来是30


# Playing video from file:
# vidcap = cv2.VideoCapture('Asiri1.mp4')
def vid_to_images(vid_file, save_folder, fpsOutput=15):
    r=fpsVideo/fpsOutput
    print("r",r)
    vidcap = cv2.VideoCapture(vid_file)
    try:
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
    except OSError:
        print ('Error: Creating directory of data')
        
    success,image = vidcap.read()
    count = 0
    success = True

    while(success):
        # Capture frame-by-frame
        success,image = vidcap.read()
        # print("image",image)
        # print("success",success)  ##测试使用
        # Saves image of the current frame in jpg file
        if(count%r==0):
            name = str(count/r) + '.jpg'
            print ('Creating...' + name)
            print("save_floder",save_folder)
            if(success):
                cv2.imwrite(os.path.join(save_folder, name), image)

        if cv2.waitKey(10) == 27:                     # exit if Escape is hit
            break

        # To stop duplicate images
        count += 1
        print("count",count)

    # When everything done, release the capture
    vidcap.release()
    cv2.destroyAllWindows()
