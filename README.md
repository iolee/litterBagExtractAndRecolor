# Cat Litter Bag Extraction and Recoloring

## Project Overview
This project automates the workflow of extracting a specific target object (a cat litter bag) from an image using a hybrid approach of instance segmentation and manual annotation refinement, followed by a recoloring process.

It utilizes a pre-trained **YOLOv8** model for initial handling and the **rembg** library for salient object detection. Crucially, it refines the extraction using specific polygon annotations (COCO format) to ensure precise edge definition—specifically keeping the "bag" while mathematically subtracting unwanted elements (like a "circle" tag).

## Workflow & Methodology
The toolchain consists of two distinct processing stages:

### 1. Precision Extraction (`BagExtract.py`)
This script combines automated background removal with strict coordinate-based masking.
- **Background Removal:** Uses the `rembg` library to strip the general background.
- **Positive Masking:** Reads `instances_default.json` to locate Category ID 2 ("bag") and enforces this area.
- **Negative Masking:** Locates Category ID 1 ("circle") and subtracts this specific polygon from the mask to ensure a clean cutout.

### 2. Post-Processing (`Recolor.py`)
- Takes the transparent, extracted output (`bag_clean.png`).
- Applies OpenCV-based color transformations to alter the product's appearance.

## File Structure

```text
.
├── BagExtract.py          # Script to extract the bag and apply polygon logic
├── Recolor.py             # Script to recolor the clean output
├── instances_default.json # COCO-format annotations (Polygon coordinates)
├── yolov8s-seg.pt         # Pre-trained YOLOv8 segmentation model weights
├── original_bag.png       # Input source image
└── README.md              # Project documentation
```
   
Note on Data Files
yolov8s-seg.pt: Weights for the YOLOv8 small segmentation model (ultralytics.nn.tasks.SegmentationModel).
instances_default.json: Contains specific polygon coordinates.
    ID 1: Circle (Area to remove).
    ID 2: Bag (Area to keep).
Prerequisites
Python 3.8+
Required Libraries
```bash
pip install rembg opencv-python numpy ultralytics
```

Usage
Step 1: Extraction
This step processes the raw image to create a transparent cutout.
  1. Ensure your input image is named original_bag.png and placed in the root directory.
  2. Run the extraction script:
```bash
python BagExtract.py
```

Output: A new file named bag_clean.png will be generated.
Step 2: Recoloring
This step applies the color transformation to the cutout.
  1. Ensure bag_clean.png exists (generated from Step 1).
  2. Run the recoloring script:
```bash
python Recolor.py
```
  3. Output: The final image will be saved (e.g., bag_recolored.png).
Configuration & Constraints
Hardcoded Categories
The BagExtract.py script is logic-bound to specific Category IDs found in the JSON file. If your JSON generation tool assigns different IDs, update these constants in the Python script:
```Python
BAG_ID = 2
CIRCLE_ID = 1
```
Image Specificity

The instances_default.json file contains hardcoded coordinates specific to the dimensions and content of original_bag.png.

Constraint: You cannot simply swap the input image without also providing a new JSON file containing the correct polygon points for the new image.

Technologies Used

Python 3.x

Ultralytics YOLOv8: For model handling and inference structure.

Rembg: For salient object background removal.

OpenCV (cv2) & NumPy: For masking logic, polygon drawing, and color space manipulation.
