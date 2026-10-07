import platform
import numpy as np

print("Machine:", platform.machine())
print("NumPy version:", np.__version__)

# A fake "image": 320 pixels tall, 320 wide, 3 color channels (RGB)
image = np.zeros((320, 320, 3), dtype=np.uint8)
print("Image shape:", image.shape)
print("Total numbers in this image:", image.size)