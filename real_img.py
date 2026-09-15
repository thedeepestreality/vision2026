import cv2
import matplotlib.pyplot as plt

image = cv2.imread("kidsnoise.bmp")

print("Shape:", image.shape)
print("Data type:", image.dtype)

plt.imshow(image)
plt.show()