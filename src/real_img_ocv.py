import cv2

image = cv2.imread("kidsnoise.bmp")

cv2.imshow("img", image)
cv2.imshow("img2", image)
cv2.waitKey(0)