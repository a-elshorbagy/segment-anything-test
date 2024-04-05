import numpy as np
import matplotlib.pyplot as plt
import cv2
import laspy
import json

import sam_wrapper as sw

mask_generator = sw.setup()

# Loading the las file from the disk
las = laspy.read("data/L1_Mission 1_refl_002.las")

# Check if the LAS file contains RGB color information
if hasattr(las, 'red') and hasattr(las, 'green') and hasattr(las, 'blue'):
    # Proceed with processing as usual
    pass
else:
    # Convert the LAS file to a format that supports RGB color information
    # For example, convert to PointFormat 3 which supports RGB colors
    converted_las = laspy.convert(las, point_format_id=3)
    # Write the converted LAS file to disk
    converted_las.write("data/temp_converted_file.las")

    # Now you can use the converted LAS file for further processing
    las = converted_las


# Transforming to a numpy array
coords = np.vstack((las.x, las.y, las.z))
point_cloud = coords.transpose()

# Gathering the colors
if hasattr(las, 'red') and hasattr(las, 'green') and hasattr(las, 'blue'):
	r=(las.red/65535*255).astype(int)
	g=(las.green/65535*255).astype(int)
	b=(las.blue/65535*255).astype(int)
	colors = np.vstack((r,g,b)).transpose()
else:
	elevation = las.z
	max_elevation = np.max(elevation)
	min_elevation = np.min(elevation)
	intensity = ((elevation - min_elevation) / (max_elevation - min_elevation)) * 255

    # Convert intensity to grayscale colors
	gray_colors = np.column_stack((intensity, intensity, intensity))

	# Update LAS object with grayscale colors
	las.red = gray_colors[:, 0]
	las.green = gray_colors[:, 1]
	las.blue = gray_colors[:, 2]

	colors = gray_colors


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

#print (result)
# Write segmentation data to JSON file
#with open("output/segmentation_masks.json", "w") as json_file:
 #   json.dump(segmentation_data, json_file)

#print("Segmentation masks exported successfully to segmentation_masks.json")


fig = plt.figure(figsize=(np.shape(spherical_image)[1]/72,
np.shape(spherical_image)[0]/72))
fig.add_axes([0,0,1,1])
plt.imshow(spherical_image)
color_mask = sw.sam_masks(result)
plt.axis('off')
plt.savefig("output/L1_Mission 1_refl_002.tiff", format='tiff')
print ("2d exported successfully")


image = cv2.imread ("output/L1_Mission 1_refl_002.tiff")
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
sw.export_point_cloud ("output/L1_Mission 1_refl_002_modified_point_cloud.las",modified_point_cloud) 
print ("point_cloud exported successfully")


