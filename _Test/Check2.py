import cv2
import numpy as np

img_rgb = cv2.imread('../Detect_Same_Images_in_Bigger_Image/mario.png')
template = cv2.imread('../Detect_Same_Images_in_Bigger_Image/mario_coin.png')
w, h = template.shape[:-1]

res = cv2.matchTemplate(img_rgb, template, cv2.TM_CCOEFF_NORMED)
threshold = .9
loc = np.where(res >= threshold)
for pt in zip(*loc[::-1]):  # Switch columns and rows
    cv2.rectangle(img_rgb, pt, (pt[0] + w, pt[1] + h), (0, 0, 255), 2)

cv2.imwrite('result.png', img_rgb)