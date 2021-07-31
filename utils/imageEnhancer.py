import cv2

clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))

def image_enhance(img):
    #-----Converting image to LAB Color model----------------------------------- 
    lab= cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    # cv2.imshow("lab",lab)

    #-----Splitting the LAB image to different channels-------------------------
    l, a, b = cv2.split(lab)
    # cv2.imshow('l_channel', l)
    # cv2.imshow('a_channel', a)
    # cv2.imshow('b_channel', b)

    #-----Applying CLAHE to L-channel-------------------------------------------

    cl = clahe.apply(l)
    # cv2.imshow('CLAHE output', cl)

    #-----Merge the CLAHE enhanced L-channel with the a and b channel-----------
    limg = cv2.merge((cl,a,b))
    # cv2.imshow('limg', limg)

    #-----Converting image from LAB Color model to RGB model--------------------
    final = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    return final

def image_sharpen(img):
    # Load the image
    # image = cv2.imread("35.0.jpg")
    # Blur the image
    gauss = cv2.GaussianBlur(img, (7,7), 0)
    # Apply Unsharp masking
    unsharp_image = cv2.addWeighted(img, 2, gauss, -1, 0)
    return unsharp_image