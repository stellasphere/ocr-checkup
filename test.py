from ocrcheckup import Benchmark
import ocrcheckup

import supervision as sv
import os

from roboflow import Roboflow

import ocrcheckup.evaluation
import ocrcheckup.models
import ocrcheckup.utils
from ocrcheckup.evaluation import StringMetrics
import numpy as np


DATASET_ID = "focused-scene-ocr"

FocusedSceneBenchmark = Benchmark.from_roboflow_dataset(
    "Focused Scene",
    api_key=os.environ["ROBOFLOW_API_KEY"],
    workspace="leo-ueno",
    project=DATASET_ID,
    version=1,
)

from ocrcheckup.benchmark.model import OCRBaseModel, OCRModelResponse, OCRModelInfo

from ocrcheckup.models import *

models = [
    DocTR_RoboflowHosted(api_key=os.environ["ROBOFLOW_API_KEY"]),
    Moondream2(),
    Claude_3_Opus(),
    Claude_3_Sonnet(),
    Claude_3_Haiku(),
    Claude_3_5_Sonnet(),
    Claude_3_5_Sonnet_V2(),
    Claude_3_5_Haiku(),
    Claude_3_7_Sonnet(),
    TrOCR(),
    Gemini_1_5_Pro(),
    Gemini_1_5_Flash(),
    Gemini_1_5_Flash_8B(),
    Gemini_2_5_Pro_Preview(),
    Gemini_2_0_Flash(),
    Gemini_2_0_Flash_Lite(),
    EasyOCR(),
    GPT_4o(),
    O1(),
    GPT_4_5_Preview(),
    GPT_4o_Mini(),
    Idefics2(),
    MistralOCR(),
    Florence2Large(),
    Florence2Base(),
]

models_to_overwrite = []

benchmark_results = FocusedSceneBenchmark.benchmark(
    models,
    autosave_dir=f"results/{DATASET_ID}",
    create_autosave=True,
    use_autosave=True, # Make sure this is True to test loading/overwriting
    run_models=True,
    overwrite=models_to_overwrite, # Pass the list or boolean here
)
print("Benchmark Results:", type(benchmark_results))

string_metrics = ocrcheckup.evaluation.StringMetrics.from_benchmark_model_results(
    benchmark_results,
    handle_empty_results="zero"
)
print("String Metrics:")
print(ocrcheckup.utils.pretty_json(string_metrics))

speed_metrics = ocrcheckup.evaluation.SpeedMetrics.from_benchmark_model_results(
    benchmark_results,
    handle_empty_results='zero'
)
print("Speed Metrics:")
print(ocrcheckup.utils.pretty_json(speed_metrics))

# --- New Analysis using indexed_results from loaded data AND current benchmark metadata --- #
print("\n--- Analyzing Poorly Performing Images (using loaded results + current benchmark metadata) ---")

# Check if benchmark_results exists and is not empty
if not benchmark_results or not isinstance(benchmark_results, list):
    print("Benchmark results are missing or invalid.")
# Check if the current FocusedSceneBenchmark object has the required data
elif not hasattr(FocusedSceneBenchmark, 'images') or not hasattr(FocusedSceneBenchmark, 'annotations') or not hasattr(FocusedSceneBenchmark, 'metadata'):
    print("Current FocusedSceneBenchmark object is missing images, annotations, or metadata.")
else:
    try:
        # --- Use CURRENT benchmark object for ground truth and metadata --- #
        num_images = len(FocusedSceneBenchmark.images)
        current_annotations = FocusedSceneBenchmark.annotations
        current_metadata = FocusedSceneBenchmark.metadata
        # --- End Use CURRENT --- #

        if num_images != len(current_annotations) or num_images != len(current_metadata):
             # This check uses the current benchmark's counts
             raise ValueError("Current benchmark image, annotation, or metadata count mismatch.")

        # Also check if the loaded results seem to match the expected number of images
        # (This is a sanity check, the primary source is the current benchmark)
        if hasattr(benchmark_results[0], 'indexed_results') and len(benchmark_results[0].indexed_results) != num_images:
             print(f"Warning: Loaded results count ({len(benchmark_results[0].indexed_results)}) doesn't match current benchmark image count ({num_images}). Analysis might be incorrect.")

    except (AttributeError, IndexError, ValueError, TypeError) as e:
        print(f"Could not get data from current FocusedSceneBenchmark object: {e}")
        num_images = 0
        current_annotations = []
        current_metadata = []

    num_models = len(benchmark_results)
    image_analysis = {} # Store dict {index: {'score': float, 'filename': str}}

    if num_images == 0 or num_models == 0:
        print("No images or models to analyze.")
    else:
        print(f"Analyzing {num_models} models across {num_images} images (using loaded results + current metadata)...")

        for i in range(num_images):
            image_levenshtein_ratios = []
            try:
                # --- Get ground truth and filename from CURRENT benchmark --- #
                ground_truth = current_annotations[i]
                image_filename = current_metadata[i].get('image_filename', f'index_{i}_missing_filename')
                # --- End Get from CURRENT --- #
            except IndexError:
                 print(f"Warning: Index {i} out of bounds for current annotations/metadata. Skipping analysis for this image.")
                 image_analysis[i] = {'score': 0.0, 'filename': f'index_{i}_missing_metadata'}
                 continue

            # --- Iterate through LOADED model results --- #
            for model_result in benchmark_results:
                try:
                    # --- Access performance from LOADED result --- #
                    response = model_result.indexed_results[i]
                    # --- End Access performance --- #

                    # --- Process the response (same logic as before) --- #
                    if response is None:
                         print(f"Warning: Missing result for image index {i} ({image_filename}) in loaded model {model_result.model.name}. Treating as failure.")
                         image_levenshtein_ratios.append(0.0)
                    elif isinstance(response, OCRModelResponse):
                        if response.success and response.prediction is not None:
                            # --- Calculate ratio using CURRENT ground truth --- #
                            ratio = StringMetrics(response.prediction, ground_truth).levenshtein_ratio()
                            # --- End Calculate ratio --- #
                            image_levenshtein_ratios.append(ratio)
                        else:
                            image_levenshtein_ratios.append(0.0)
                    else:
                         print(f"Warning: Invalid data type found at index {i} ({image_filename}) for loaded model {model_result.model.name}. Type: {type(response)}. Treating as failure.")
                         image_levenshtein_ratios.append(0.0)
                    # --- End Process response --- #

                except IndexError:
                    print(f"Warning: Index {i} out of bounds for loaded indexed_results in model {model_result.model.name} (Size: {len(model_result.indexed_results)}) for image {image_filename}. Treating as failure.")
                    image_levenshtein_ratios.append(0.0)
                except AttributeError:
                     print(f"Warning: Loaded result for model {model_result.model.name} seems malformed (missing indexed_results?) for image {i} ({image_filename}). Treating as failure.")
                     image_levenshtein_ratios.append(0.0)
                except Exception as e:
                     print(f"Error accessing loaded result for image {i} ({image_filename}), model {model_result.model.name}: {e}")
                     image_levenshtein_ratios.append(0.0)
            # --- End Iterate through LOADED model results --- #

            # Calculate the average score for the image
            if image_levenshtein_ratios:
                average_score = sum(image_levenshtein_ratios) / len(image_levenshtein_ratios)
            else:
                average_score = 0.0

            # Store score and filename (from CURRENT benchmark)
            image_analysis[i] = {'score': average_score, 'filename': image_filename}


        # Sort based on score
        sorted_analysis = sorted(image_analysis.items(), key=lambda item: item[1]['score'])

        # Get the top 5 worst performing filenames
        top_5_worst_files = [item[1]['filename'] for item in sorted_analysis[:5]]

        print("\nTop 5 Image Filenames with Poorest Average Performance (Lowest Levenshtein Ratio):")
        if not top_5_worst_files:
             print("Could not determine worst performing images (check analysis details).")
        else:
            for filename in top_5_worst_files:
                 print(f"- {filename}")

            # Optional: Print scores and filenames
            print("\nImage Filename                 | Average Levenshtein Ratio")
            print("-----------------------------|---------------------------")
            for index, data in sorted_analysis[:5]:
                print(f"{data['filename']:<28} | {data['score']:.4f}")

            # --- Show Predictions for Top 5 Worst Images --- #
            print("\n--- Predictions for Top 5 Worst Performing Images --- ")
            # Use top_5_worst_items which contains (index, data) tuples
            top_5_worst_items = sorted_analysis[:5]
            for index, data in top_5_worst_items:
                filename = data['filename']
                score = data['score']
                print(f"\n--------------------------------------------------")
                print(f"Image: {filename} (Average Score: {score:.4f})")
                try:
                     # Use current_annotations from the outer scope
                     ground_truth = current_annotations[index]
                     print(f"Ground Truth: {ground_truth}")
                except IndexError:
                     print("Ground Truth: Error retrieving!")
                except NameError:
                     print("Ground Truth: Error retrieving (current_annotations not defined)!")


                print("Model Predictions:")
                # Loop through the loaded benchmark_results
                for model_result in benchmark_results:
                     try:
                         model_name = model_result.model.name
                     except AttributeError:
                         model_name = "[Unknown Model Name]"

                     try:
                         # Access the specific result for this image index
                         response = model_result.indexed_results[index]

                         # Determine what to print based on the response
                         if response is None:
                              prediction_text = "[Result Missing]"
                         elif not isinstance(response, OCRModelResponse):
                              prediction_text = f"[Invalid Result Type: {type(response)}]"
                         elif response.success:
                              # Handle potential None prediction even on success
                              prediction_text = response.prediction if response.prediction is not None else "[Prediction is None]"
                         else:
                              # Include error message for failed predictions
                              error_msg = response.error_message if response.error_message else "No error message"
                              prediction_text = f"[FAILED: {error_msg}]"

                     except IndexError:
                         prediction_text = "[Index Error accessing result]"
                     except AttributeError:
                         prediction_text = "[Attribute Error accessing result (maybe missing indexed_results)]"
                     except Exception as e:
                          prediction_text = f"[Unexpected error retrieving prediction: {e}]"

                     # Print model name and its prediction/status
                     print(f"  - {model_name}:")
                     # Indent prediction text for readability
                     print(f"    {prediction_text}") # Simple string representation

            print(f"--------------------------------------------------")
            # --- End Show Predictions --- #
