import numpy as np
import matplotlib.pyplot as plt
import cv2
import laspy

import sam_wrapper as sw

mask_generator = sw.setup()

# Loading the las file from the disk
las = laspy.read("data/ITC_BUILDING.las")

# Transforming to a numpy array
coords = np.vstack((las.x, las.y, las.z))
point_cloud = coords.transpose()

# Gathering the colors
r=(las.red/65535*255).astype(int)
g=(las.green/65535*255).astype(int)
b=(las.blue/65535*255).astype(int)
colors = np.vstack((r,g,b)).transpose()
resolution = 500

# Defining the position in the point cloud to generate a panorama
center_coordinates = [189, 60, 2]

# Function Execution
spherical_image, mapping = sw.generate_spherical_image(center_coordinates, point_cloud, colors, resolution)

'''
# Plotting with matplotlib
fig = plt.figure(figsize=(np.shape(spherical_image)[1]/72,
np.shape(spherical_image)[0]/72))
fig.add_axes([0,0,1,1])
plt.imshow(spherical_image)
plt.axis('off')

# Saving to the disk
plt.savefig("output/ITC_BUILDING_spherical_projection.jpg")

image_bgr = cv2.imread("output/ITC_BUILDING_spherical_projection.jpg")
image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
#t0 = time.time()
'''

result = mask_generator.generate(spherical_image)
#t1 = time.time()

fig = plt.figure(figsize=(np.shape(spherical_image)[1]/72,
np.shape(spherical_image)[0]/72))
fig.add_axes([0,0,1,1])
plt.imshow(spherical_image)
color_mask = sw.sam_masks(result)
plt.axis('off')
plt.savefig("output/ITC_BUILDING_spherical_projection_segmented.jpg")
print ("2d exported successfully")


image = cv2.imread ("output/ITC_BUILDING_spherical_projection_segmented.tiff")
print ("image read successfully")
point_cloud = point_cloud

modified_point_cloud= sw.color_point_cloud(image, point_cloud, mapping)
print ("point_cloud colored successfully")
'''

point_cloud_result = laspy.LasData()
point_cloud_result.copy_header(modified_point_cloud)
point_cloud_result.points = modified_point_cloud.points
point_cloud_result.write("output/ITC_BUILDING_modified_point_cloud.las")
print ("3d exported successfully")
'''
sw.export_point_cloud ("output/ITC_BUILDING_modified_point_cloud.las",modified_point_cloud) 
print ("point_cloud exported successfully")


