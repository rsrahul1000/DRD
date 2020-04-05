import cv2
import numpy as np

def MA(base_path, image_id):
    eye = cv2.imread(base_path + image_id)

    image = cv2.cvtColor(eye, cv2.COLOR_BGR2RGB)
    median = cv2.medianBlur(image, 1)

    green_image = median.copy()  # Make a copy
    green_image[:, :, 0] = 0
    green_image[:, :, 2] = 0
    gPixels = np.array(green_image)

    # -----Converting image to LAB Color model-----------------------------------
    lab = cv2.cvtColor(gPixels, cv2.COLOR_BGR2LAB)

    # -----Splitting the LAB image to different channels-------------------------
    l, a, b = cv2.split(lab)

    # -----Applying CLAHE to L-channel-------------------------------------------
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)

    # -----Merge the CLAHE enhanced L-channel with the a and b channel-----------
    limg = cv2.merge((cl, a, b))

    # -----Converting image from LAB Color model to RGB model--------------------
    final = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

    #############################################################################
    edges = cv2.Canny(final, 70, 35)
    final_gray = cv2.cvtColor(final, cv2.COLOR_BGR2GRAY)

    edge_test = final_gray + edges

    eye_final = edge_test

    # Perform closing to find individual objects
    eye_final = cv2.dilate(eye_final, (3, 3), iterations=2)  # eye_final.dilate(2)
    eye_final = cv2.erode(eye_final, (3, 3), iterations=1)

    eye_final = cv2.dilate(eye_final, (3, 3), iterations=4)
    eye_final = cv2.erode(eye_final, (3, 3), iterations=3)

    # Setup SimpleBlobDetector parameters.
    params = cv2.SimpleBlobDetector_Params()

    # Set Area filtering parameters
    params.filterByArea = True
    params.minArea = 10  # lesser the area more the number of keypoints generated

    # Detect blobs.
    detector = cv2.SimpleBlobDetector_create(params)
    big_blobs = detector.detect(eye_final)

    # create a blank image to mask
    blank = np.zeros(eye.shape, np.uint8)
    blobs = cv2.drawKeypoints(image, big_blobs, blank, (255, 255, 255),
                              cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    blobs_gray = cv2.cvtColor(blobs, cv2.COLOR_RGB2GRAY)
    eye_final = eye_final - blobs_gray
    eye_final = cv2.erode(eye_final, (3, 3), iterations=1)

    print(eye_final[100, 200])
    print(eye_final[200, 300])
    print(eye_final[299, 300])

    # Setup SimpleBlobDetector parameters.
    params = cv2.SimpleBlobDetector_Params()
    # Change thresholds
    params.minThreshold = 1;
    params.maxThreshold = 500;

    # Detect blobs.
    detector = cv2.SimpleBlobDetector_create(params)
    # Detect blobs.
    small_blobs = detector.detect(eye_final)
    #print(small_blobs)
    if small_blobs:

        small_blobs_detected = cv2.drawKeypoints(image, big_blobs, blank, (255, 255, 255),
                                                 cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        eye_final_small = eye_final - cv2.cvtColor(small_blobs_detected, cv2.COLOR_RGB2GRAY)
        # print("secondtime")
        print(eye_final[100, 200])
        print(eye_final[200, 300])
        print(eye_final[299, 300])

        return eye_final_small, blobs

    return 'np'

def extract_bv(base_path, image_id):
    image = cv2.imread(base_path + image_id)
    b, green_fundus, r = cv2.split(image)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    contrast_enhanced_green_fundus = clahe.apply(green_fundus)

    # applying alternate sequential filtering (3 times closing opening)
    r1 = cv2.morphologyEx(contrast_enhanced_green_fundus, cv2.MORPH_OPEN,
                          cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)), iterations=1)
    R1 = cv2.morphologyEx(r1, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)), iterations=1)
    r2 = cv2.morphologyEx(R1, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11)), iterations=1)
    R2 = cv2.morphologyEx(r2, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11)), iterations=1)
    r3 = cv2.morphologyEx(R2, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (23, 23)), iterations=1)
    R3 = cv2.morphologyEx(r3, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (23, 23)), iterations=1)
    f4 = cv2.subtract(R3, contrast_enhanced_green_fundus)
    f5 = clahe.apply(f4)

    # removing very small contours through area parameter noise removal
    ret, f6 = cv2.threshold(f5, 15, 255, cv2.THRESH_BINARY)
    mask = np.ones(f5.shape[:2], dtype="uint8") * 255
    im2, contours, hierarchy = cv2.findContours(f6.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        if cv2.contourArea(cnt) <= 200:
            cv2.drawContours(mask, [cnt], -1, 0, -1)
    im = cv2.bitwise_and(f5, f5, mask=mask)
    ret, fin = cv2.threshold(im, 15, 255, cv2.THRESH_BINARY_INV)
    newfin = cv2.erode(fin, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=1)

    # removing blobs of unwanted bigger chunks taking in consideration they are not straight lines like blood
    # vessels and also in an interval of area
    fundus_eroded = cv2.bitwise_not(newfin)
    xmask = np.ones(image.shape[:2], dtype="uint8") * 255
    x1, xcontours, xhierarchy = cv2.findContours(fundus_eroded.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in xcontours:
        shape = "unidentified"
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.04 * peri, False)
        if len(approx) > 4 and cv2.contourArea(cnt) <= 3000 and cv2.contourArea(cnt) >= 100:
            shape = "circle"
        else:
            shape = "veins"
        if (shape == "circle"):
            cv2.drawContours(xmask, [cnt], -1, 0, -1)

    finimage = cv2.bitwise_and(fundus_eroded, fundus_eroded, mask=xmask)
    blood_vessels = cv2.bitwise_not(finimage)
    return blood_vessels


def exudate(base_path, image_id):
    img = cv2.imread(base_path + image_id)
    jpegImg = 0
    grayImg = 0
    curImg = 0

    curImg = np.array(img)  ##Convert jpegFile to numpy array (Required for CV2)

    print(curImg.shape)

    gcImg = curImg[:, :, 1]
    curImg = gcImg

    clahe = cv2.createCLAHE()
    clImg = clahe.apply(curImg)
    #   clImg = clahe.apply(clImg)
    curImg = clImg

    # Creating Structurig Element
    strEl = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (6, 6))
    # Dilation
    dilateImg = cv2.dilate(curImg, strEl)
    curImg = dilateImg

    # Thresholding with Complement/15
    retValue, threshImg = cv2.threshold(curImg, 235, 247, cv2.THRESH_BINARY_INV)
    curImg = threshImg

    # Median Filtering
    medianImg = cv2.medianBlur(curImg, 3)
    curImg = medianImg
    return curImg


def haemorrhage(base_path, image_id):
    image = cv2.imread(base_path + image_id)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    median = cv2.medianBlur(image, 1)
    # compare = np.concatenate((image, median), axis=1) #side by side comparison
    medianPixels = np.array(median)

    b, g, r = cv2.split(median)
    green_image = median.copy()  # Make a copy
    green_image[:, :, 0] = 0
    green_image[:, :, 2] = 0

    gPixels = np.array(green_image)
    # -----Converting image to LAB Color model-----------------------------------
    lab = cv2.cvtColor(gPixels, cv2.COLOR_BGR2LAB)
    cv2.imshow("lab", lab)

    # -----Splitting the LAB image to different channels-------------------------
    l, a, b = cv2.split(lab)

    # -----Applying CLAHE to L-channel-------------------------------------------
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)

    # -----Merge the CLAHE enhanced L-channel with the a and b channel-----------
    limg = cv2.merge((cl, a, b))

    # -----Converting image from LAB Color model to RGB model--------------------
    final = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

    # Gaussian matched filtering method  is applied for further vessel enhancement
    height, width = final.shape[0:2]
    rotationMatrix = cv2.getRotationMatrix2D((width / 2, height / 2), 15, 7)

    blur = cv2.medianBlur(final, 7, 0)

    ret, thresh2 = cv2.threshold(cv2.cvtColor(blur, cv2.COLOR_RGB2GRAY), 0, 255,
                                 cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    opening = cv2.morphologyEx(thresh2, cv2.MORPH_OPEN, (10, 10))

    return opening