import numpy as np
import cv2

import sam_wrapper as sw

mask_generator, pridector = sw.setup()

image_bgr = cv2.imread("/Users/aelshorbagy/Documents/GitHub/segment-anything-test/data/Ashraf/IMG_1199.jpeg")
image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
result = mask_generator.generate(image_rgb)

sw.plot_image(image_rgb, result, "/Users/aelshorbagy/Documents/GitHub/segment-anything-test/data/Ashraf/SAM_IMG_1199.jpeg")


