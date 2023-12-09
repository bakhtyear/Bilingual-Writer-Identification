import os

import cv2

folder = "Bangla"
i = 0
for file in os.listdir(folder):
    i += 1
    image = os.path.join(folder, file)

    im = cv2.imread(image)[700:, :]
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (25, 25), 0)

    ret, thresh = cv2.threshold(blur, 220, 255, 1)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    print("Number of Contours found = " + str(len(contours)))

    idx = 0
    for cnt in contours:
        idx += 1
        x, y, w, h = cv2.boundingRect(cnt)
        roi = im[y:y + h, x:x + w]
        segment = x//100+(y//100)*10
        # cv2.imwrite(f'{idx}.jpg', roi)
        # cv2.rectangle(im,(x,y),(x+w,y+h),(200,0,0),2)


        roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        roi_blur = cv2.GaussianBlur(roi_gray, (5, 5), 0)

        ret, roi_thresh = cv2.threshold(roi_blur, 230, 255, 0)
        # roi_thresh = cv2.adaptiveThreshold(
        #     src=roi_blur,
        #     maxValue=255,
        #     adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        #     thresholdType=cv2.THRESH_BINARY,
        #     blockSize=35,
        #     C=2
        # )
        # cv2.imshow("ROI", roi_thresh)
        roi_contours, hierarchy = cv2.findContours(
            image=roi_thresh,
            mode=cv2.RETR_LIST,
            method=cv2.CHAIN_APPROX_NONE,
        )

        for cnt2 in roi_contours:
            x, y, w, h = cv2.boundingRect(cnt2)
            if 190 > h > 170 and 190 > w > 170:
                roi2 = roi[y:y + h, x:x + w]
                folder_path = f'Outputs/ID {i:3d} [{file}]'
                os.makedirs(folder_path, exist_ok=True)
                cv2.imwrite(f"{folder_path}/{segment},{y // 100},{x // 100}.jpg", roi2)
                # cv2.rectangle(roi,(x,y),(x+w,y+h),(255,0,0),2)
