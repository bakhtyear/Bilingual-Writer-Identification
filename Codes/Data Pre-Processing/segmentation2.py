import os
import numpy as np

import cv2


def find_center(i, x_og, y_og):
    b = cv2.blur(i, (5, 5), 0)
    _, t = cv2.threshold(b, 128, 255, 0)
    c, _ = cv2.findContours(t, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    for cnt in c:
        x, y, w, h = cv2.boundingRect(cnt)
        if 85 < w < 95 and 85 < h < 95:
            return x_og + x + w//2, y_og + y + h//2


folder = "English"
for idx, file in enumerate(os.listdir(folder)):
    image = os.path.join(folder, file)

    im = cv2.imread(image)
    centers = []
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    corner = cv2.rectangle(gray, (100,300),(2450,3100),(0,0,0),-1)
    blur = cv2.blur(corner, (27, 27), 0)
    _, thresh = cv2.threshold(blur, 150, 255, 0)
    blur = cv2.blur(thresh, (27, 27), 0)
    _, thresh = cv2.threshold(blur, 130, 255, 1)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if 110 < w < 134 and 110 < h < 134:
            centers.append(find_center(corner[y:y + h, x:x + w], x, y))
    if len(centers) != 4 or None in centers:
        print(centers)
        cv2.imshow("Error", im)
        cv2.waitKey(0)
        continue
    centers = sorted(centers, key=lambda x: x[0] * x[1])
    src_pts = np.array(centers, dtype=np.float32)
    width = 2481
    height = 3507

    dst_pts = np.array([
        (120, 120),
        (width-120, 120),
        (120, height-120),
        (width-120, height-120)],
        dtype=np.float32
    )
    matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)

    output_image = cv2.warpPerspective(im, matrix, (width, height))
    cv2.imwrite('asw.jpg', output_image)

    dx = 295
    cx = 63
    dy = 381
    cy = 203

    for i in range(8):
        for j in range(8):
            roi = output_image[i*dy+cy+70:(i+1)*dy+cy-10, j*dx+cx+10:(j+1)*dx+cx-10]
            cv2.imshow("test", roi)
            cv2.waitKey(2)
            # exit()
            folder_path = f'Outputs/English/ID {idx:3d} [{file}]'
            os.makedirs(folder_path, exist_ok=True)
            cv2.imwrite(f"{folder_path}/{i*8+j}.jpg", roi)