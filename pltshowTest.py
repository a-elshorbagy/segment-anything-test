import numpy as np
import matplotlib.pyplot as plt
import cv2
import laspy

las = laspy.read("data/ITC_BUILDING.las")

# Extract x, y, and z coordinates
x = las.x
y = las.y
z = las.z

# Create a 3D scatter plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x, y, z, s=1)  # Adjust 's' for point size if needed

# Set labels and title
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Point Cloud Visualization')

# Show plot
plt.show()
print ('plot drawn successfuly')