import sys
sys.path.append ("/Users/aelshorbagy/Documents/GitHub/segment-anything-test/GroundingDINO")
import Annotate_Image as ani
import cv2

device, model = ani.setup()

IMAGE_PATH = "/Users/aelshorbagy/Documents/GitHub/segment-anything-test/data/Ashraf/IMG_1199.jpeg"
TEXT_PROMPT = "sidewalk . sign. crack ."
BOX_THRESHOLD = 0.35
TEXT_THRESHOLD = 0.25

boxes, annotated_frame = ani.annotate_image_GDINO (model,IMAGE_PATH, TEXT_PROMPT, BOX_THRESHOLD, TEXT_THRESHOLD, device)
 
# Save the annotated image
cv2.imwrite("/Users/aelshorbagy/Documents/GitHub/segment-anything-test/output/Ashraf/annotated_image.jpg", annotated_frame)