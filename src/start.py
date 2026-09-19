import numpy as np
import matplotlib.pyplot as plt

# image = np.array([
#     [0,   50,  100, 150],
#     [50,  100, 150, 200],
#     [100, 150, 200, 255],
#     [150, 200, 255, 255]
# ], dtype=np.uint8)

image = np.zeros((480,640), dtype=np.uint8)
image[10, 20] = 255
image[100:200,100:300] = 255

print(image)
print("Shape:", image.shape)
print("Data type:", image.dtype)

plt.imshow(image, cmap="gray", vmin=0, vmax=255)
# plt.colorbar()
plt.show()



# Image Processing (Image Signal Processing, ISP)
# Image -> Image

# Computer Vision
# Image -> Data