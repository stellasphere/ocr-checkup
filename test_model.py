import os
import cv2
from dotenv import load_dotenv

# Import all models from the models module
from ocrcheckup.models import *
from ocrcheckup import utils

# Load environment variables (e.g., for API keys)
load_dotenv()

# --- Configuration ---
# Select a test image from your dataset
# Using one found in 'datasets/industrial-focus-scene/test'
TEST_IMAGE_PATH = "datasets/industrial-focus-scene/test/chrome_8v7GsBtbvv_png.rf.5609cffeb8b1e57995cb64d35d94474b.jpg"

# Optional: Estimate compute cost per second for the model
# If you have an idea of how much your compute costs (e.g., $0.001/sec), set it here.
# Otherwise, set to None and cost will not be calculated automatically.
# Note: Some models might have their own cost calculation mechanisms.
COMPUTE_COST_PER_SEC = None # Example: 0.001

# --- Model Selection & Initialization ---
# !!! CHANGE THIS LINE TO TEST A DIFFERENT MODEL !!!
# Examples:
# model_to_test = Moondream2(cost_per_second=COMPUTE_COST_PER_SEC)
# model_to_test = DocTR_RoboflowHosted(api_key=os.environ.get("ROBOFLOW_API_KEY"))
# model_to_test = GPT_4o(api_key=os.environ.get("OPENAI_API_KEY"), cost_per_second=COMPUTE_COST_PER_SEC)
# model_to_test = TrOCR()
# model_to_test = EasyOCR()
# model_to_test = Idefics2()
model_to_test = Gemini_1_5_Pro()

# --- Initialization & Testing ---
model_name = model_to_test.info().name if hasattr(model_to_test, 'info') else "Selected Model"
print(f"Loading {model_name} model...")

# Instantiate the selected model
# (Already done above in the selection part)

# Test if the model initializes correctly (optional but recommended)
if hasattr(model_to_test, 'test') and not model_to_test.test():
    print(f"{model_name} model failed initialization test. Exiting.")
    exit()
elif not hasattr(model_to_test, 'test'):
     print(f"Note: {model_name} does not have a '.test()' method implemented.")


print("\nLoading test image...")
# Load the single test image using OpenCV
image = cv2.imread(TEST_IMAGE_PATH)

if image is None:
    print(f"Error: Could not load image at {TEST_IMAGE_PATH}")
    exit()

# Ensure image is in RGB format (OpenCV loads as BGR by default)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# --- Evaluation ---
print(f"\nRunning evaluation with {model_name} on image: {TEST_IMAGE_PATH}")

# Use the run_for_eval method which handles timing and potential errors
response = model_to_test.run_for_eval(image_rgb)

# --- Results ---
print(f"\n--- {model_name} Evaluation Results ---")
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