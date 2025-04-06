import os
import cv2
from dotenv import load_dotenv

from ocrcheckup.models import Moondream2
from ocrcheckup import utils

# Load environment variables (e.g., for API keys if needed, though not for Moondream2)
load_dotenv()

# --- Configuration ---
# Select a test image from your dataset
# Using one found in 'datasets/industrial-focus-scene/test'
TEST_IMAGE_PATH = "datasets/industrial-focus-scene/test/chrome_8v7GsBtbvv_png.rf.5609cffeb8b1e57995cb64d35d94474b.jpg"

# Optional: Estimate compute cost per second for Moondream2
# If you have an idea of how much your compute costs (e.g., $0.001/sec), set it here.
# Otherwise, set to None and cost will not be calculated automatically.
COMPUTE_COST_PER_SEC = None # Example: 0.001

# --- Initialization ---
print(f"Loading Moondream2 model...")
# Instantiate the Moondream2 model
moondream_model = Moondream2(cost_per_second=COMPUTE_COST_PER_SEC)

# Test if the model initializes correctly (optional but recommended)
if not moondream_model.test():
    print("Moondream2 model failed initialization test. Exiting.")
    exit()

print("\nLoading test image...")
# Load the single test image using OpenCV
image = cv2.imread(TEST_IMAGE_PATH)

if image is None:
    print(f"Error: Could not load image at {TEST_IMAGE_PATH}")
    exit()

# Ensure image is in RGB format (OpenCV loads as BGR by default)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# --- Evaluation ---
print(f"\nRunning evaluation on image: {TEST_IMAGE_PATH}")

# Use the run_for_eval method which handles timing and potential errors
response = moondream_model.run_for_eval(image_rgb)

# --- Results ---
print("\n--- Evaluation Results ---")
if response.success:
    print(f"Prediction:\n{response.prediction}")
    print(f"\nElapsed Time: {response.elapsed_time:.4f} seconds")
    if response.cost is not None:
        print(f"Estimated Cost: ${response.cost:.6f}")
    else:
        print("Cost: Not calculated (cost_per_second not provided or model doesn't report cost)")
else:
    print(f"Evaluation Failed!")
    print(f"Error Message: {response.error_message}")
    print(f"Elapsed Time: {response.elapsed_time:.4f} seconds (until failure)")

print("\nScript finished.") 