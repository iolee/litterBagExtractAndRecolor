Cat Litter Bag Extraction and Recoloring
This project automates the process of extracting a specific object (a cat litter bag) from an image using instance segmentation and then recoloring it. It utilizes a pre-trained YOLOv8 model for initial detection, refined by specific polygon annotations, and a recoloring script for post-processing.


Overview
This toolchain consists of two main scripts:

BagExtract.py: Removes the background from an input image (original_bag.png). It uses the rembg library for salient object detection and refines the result using specific segmentation coordinates provided in instances_default.json (specifically for the "bag" class). It also handles the removal of specific unwanted elements (like the "circle" class).

Recolor.py: Takes the clean, extracted image (bag_clean.png) and applies a color transformation to change the bag's appearance.


Prerequisites
Python 3.8+


Required Python Libraries:

rembg

opencv-python (cv2)

numpy

ultralytics (for YOLOv8 model handling)


File Structure
Ensure your project directory is organized as follows:

.

├── BagExtract.py            # Script to extract the bag and remove background

├── Recolor.py               # Script to recolor the extracted bag

├── instances_default.json   # COCO-format annotations for segmentation coordinates

├── yolov8s-seg.pt           # Pre-trained YOLOv8 segmentation model weights

├── original_bag.png         # Input image to be processed

└── README.md                # Project documentation


Note on Data Files:


yolov8s-seg.pt: This is the weights file for the YOLOv8 small segmentation model. It identifies the model type (ultralytics.nn.tasks.SegmentationModel) and contains the neural network layers required for inference.


instances_default.json: This JSON file contains the specific polygon coordinates for the image. It defines two categories: ID 1 (circle) and ID 2 (bag). The script uses these exact coordinates to create the final mask.

Installation

Clone the repository (if applicable) or download the source files.


Install dependencies:

pip install rembg opencv-python numpy ultralytics



Usage

Step 1: Extraction (BagExtract.py)

This script combines a pre-trained background removal model with specific annotation data to create a clean cutout of the bag.

Place your input image as original_bag.png in the root directory.

Run the script:

python BagExtract.py

Process:

The script first uses rembg to remove the general background.

It reads instances_default.json to find the exact segmentation polygon for Category ID 2 ("bag").

It masks the image to keep only the bag.

It looks for Category ID 1 ("circle") annotations and subtracts (removes) that area from the mask.

Output: 

The script generates bag_clean.png.


Step 2: Recoloring (Recolor.py)

This script applies color adjustments to the extracted image.

Ensure bag_clean.png exists (created by the previous step).

Run the script:

python Recolor.py

Output: 

The final recolored image will be saved (e.g., bag_recolored.png).



Configuration

Category IDs: The extraction script is hardcoded to look for specific IDs in the JSON file:

BAG_ID = 2

CIRCLE_ID = 1 

If your instances_default.json uses different IDs, update these constants in BagExtract.py.



Segmentation Coordinates: 

The instances_default.json file contains hardcoded coordinates specific to original_bag.png. If you use a different input image, you must generate a new JSON annotation file with the correct polygon points for that new image.
