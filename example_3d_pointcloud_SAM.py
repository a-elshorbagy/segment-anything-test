import sys
import numpy as np
import cv2

import sam_wrapper as sw

argv = sys.argv
inputUrl = None
mask_generator = None

try:
	argv = argv[argv.index("--") + 1:] # get all args after "--"
	modelType = argv[0]
	inputUrl = argv[1]
	mask_generator = sw.setup(modelType)
except:
	print("Using test defaults.")
	inputUrl = "data/ITC_BUILDING_small.ply"
	mask_generator = sw.setup()

point_cloud, colors = sw.import_point_cloud(inputUrl)

spherical_image, mapping = sw.generate_spherical_image(point_cloud, colors)

spherical_image_rgb = cv2.cvtColor(spherical_image, cv2.COLOR_BGR2RGB)
cv2.imwrite("output/spherical_projection_preview.jpg", spherical_image_rgb)
print("Saved spherical projection preview.")

result = mask_generator.generate(spherical_image)

imgUrl = "output/spherical_projection_segmented.jpg"

sw.plot_image(spherical_image, result, imgUrl)

image = cv2.imread(imgUrl)

modified_point_cloud = sw.color_point_cloud(image, point_cloud, mapping)

sw.export_point_cloud("output/pcd_results.ply", modified_point_cloud)
