import numpy as np
import cv2
import sam_wrapper as sw
import torch
import sys
sys.path.append ("/Users/aelshorbagy/Documents/GitHub/segment-anything-test/GroundingDINO")
import Annotate_Image as ani
import sam_wrapper as sw
from segment_anything import SamPredictor

#intializing GroundingDINO
device, model = ani.setup()

#intializing SAM
mask_generator, predictor = sw.setup()


#Define image path, text prompt, and thresholds
IMAGE_PATH = "/Users/aelshorbagy/Documents/GitHub/segment-anything-test/data/JPG/IMG_1199.jpeg"
TEXT_PROMPT = "sidewalk . sign. crack ."
BOX_THRESHOLD = 0.35
TEXT_THRESHOLD = 0.25

#define output path
prediction_path = "/Users/aelshorbagy/Documents/GitHub/segment-anything-test/output/JPG/annotated2_image.jpg"
annotated_image_path = "/Users/aelshorbagy/Documents/GitHub/segment-anything-test/output/JPG/annotated_Image.jpg"

#load image and define W,H
image_data = cv2.imread(IMAGE_PATH)
W, H = image_data.shape[1], image_data.shape[0]

# Check if the image is loaded as a NumPy array
if isinstance(image_data, np.ndarray):
    print("Image is loaded as a NumPy array.")
    print("Image dtype:", image_data.dtype)
else:
    print("Image is not loaded as a NumPy array.")
print ("Image shape", image_data.shape)

#use GroundingDino to annotate image for the above prompt

boxes, annotated_frame = ani.annotate_image_GDINO (model,IMAGE_PATH, TEXT_PROMPT, BOX_THRESHOLD, TEXT_THRESHOLD, device)

#Save the annotated image
cv2.imwrite(annotated_image_path, annotated_frame)
print("All bounding boxes:", boxes)

#use boxes as an input to SAM 
predictor.set_image(image_data)
print ("image loaded to predictor")

all_masks =[]
all_scores = []

for index, bbox in enumerate(boxes):
            # Preprocess bounding box
            box = torch.Tensor(bbox) * torch.Tensor([W, H, W, H])
            box[:2] -= box[2:] / 2
            box[2:] += box[:2]
            x0, y0, x1, y1 = box.int().tolist()
            # Prepare SAM prompt
            input_box = np.array([x0, y0, x1, y1])
            print ("SAM box prompt populated", index)
            input_point = None
            input_label = None
            segment_prompt = [input_point, input_label, input_box]
            # Segment using the prepared prompt
            masks, scores = sw.visualize_and_save_segmentation(segment_prompt, predictor, index, image_data, prediction_path)
            print ("SAM mask created", index)
            all_masks.extend(masks)
            all_scores.extend(scores)
all_masks_adj = np.array(all_masks)
all_scores_adj = np.array(all_scores)

sw.save_segementation(all_masks_adj,all_scores_adj,image_data, "/Users/aelshorbagy/Documents/GitHub/segment-anything-test/output/jpg/SAM_IMG_1199.jpeg")

