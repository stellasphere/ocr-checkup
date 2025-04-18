from ocrcheckup import Benchmark
import ocrcheckup

import supervision as sv
import os
import csv
from pathlib import Path

from roboflow import Roboflow

import ocrcheckup.evaluation
import ocrcheckup.models
import ocrcheckup.utils
from ocrcheckup.evaluation import StringMetrics
import ocrcheckup.evaluation as eval_utils
import numpy as np
import Levenshtein

from ocrcheckup.cost import ModelCostCalculator, ModelCost, CostType

DATASET_ID = "focused-scene-ocr"

FocusedSceneBenchmark = Benchmark.from_roboflow_dataset(
    "Focused Scene",
    api_key=os.environ["ROBOFLOW_API_KEY"],
    workspace="leo-ueno",
    project=DATASET_ID,
    version=1,
)

FocusedSceneBenchmark.showcase()

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
    Qwen_2_5_VL_7B(),
]

models_to_overwrite = []

benchmark_results = FocusedSceneBenchmark.benchmark(
    models,
    autosave_dir=f"results/{DATASET_ID}",
    create_autosave=True,
    create_autosave_with_fails=True,
    autosave_fail_threshold=0.5,
    use_autosave=True, # Make sure this is True to test loading/overwriting
    run_models=True,
    overwrite=models_to_overwrite, # Pass the list or boolean here
    time_between_runs=1
)
print("Benchmark Results:", type(benchmark_results))

# --- Prepare for CSV Export ---
csv_output_dir = Path(f"results/{DATASET_ID}_inference_details")
csv_output_dir.mkdir(parents=True, exist_ok=True)
csv_file_path = csv_output_dir / "detailed_results.csv"

# --- Initialize Cost Calculator (reuse from later in the script) ---
runtime_cost_per_second_placeholder = 0.001506661
try:
    cost_calculator_for_csv = ModelCostCalculator.default(runtime_cost_per_second=runtime_cost_per_second_placeholder)
except (ImportError, ValueError) as e:
    print(f"Failed to initialize ModelCostCalculator for CSV: {e}. Individual costs might be missing.")
    cost_calculator_for_csv = None

# --- Write Detailed CSV ---
print(f"\nWriting detailed inference results to {csv_file_path}...")
try:
    # Access ground truths (annotations) and metadata for image IDs
    ground_truths = FocusedSceneBenchmark.annotations # Use .annotations
    metadata_list = FocusedSceneBenchmark.metadata    # Use .metadata

    # Extract image filenames from metadata, providing a fallback
    if metadata_list and all('image_filename' in item for item in metadata_list):
        image_ids = [item['image_filename'] for item in metadata_list]
    else:
        print("Warning: Could not extract 'image_filename' from all metadata items. Falling back to index-based IDs.")
        # Generate index IDs if metadata is missing or incomplete
        image_ids = [f"index_{i}" for i in range(len(ground_truths))]

    if len(image_ids) != len(ground_truths):
        print(f"Warning: Mismatch between number of extracted image IDs ({len(image_ids)}) and annotations ({len(ground_truths)}). CSV data might be misaligned.")
        # Adjust image_ids length to match ground_truths to prevent IndexError, though data might be wrong
        image_ids = [image_ids[i] if i < len(image_ids) else f"missing_id_{i}" for i in range(len(ground_truths))]


    with open(csv_file_path, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = [
            'model_name', 'model_version', 'image_id', 'ground_truth',
            'prediction', 'success', 'error_message', 'start_time',
            'elapsed_time', 'cost_usd', 'cost_details_repr',
            'levenshtein_distance', 'levenshtein_ratio', 'is_correct'
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        # Iterate through results for each model
        for model_result in benchmark_results:
            model_info = model_result.model
            model_name = model_info.name
            model_version = model_info.version

            # Iterate through individual responses for the model
            for i, response in enumerate(model_result.indexed_results):
                # Handle potential None responses if benchmark allows it
                if response is None:
                    print(f"Warning: Skipping None response for model {model_name}, image index {i}")
                    # Optionally write a row indicating failure if needed
                    # writer.writerow({'model_name': model_name, 'model_version': model_version, 'image_id': image_ids[i] if i < len(image_ids) else f"index_{i}", 'success': False, 'error_message': 'Benchmark returned None response'})
                    continue

                # Defensive checks for index boundaries
                if i >= len(image_ids) or i >= len(ground_truths):
                     print(f"Warning: Skipping response index {i} for model {model_name} due to index out of bounds for image_ids/ground_truths.")
                     continue

                image_id = image_ids[i]
                gt = ground_truths[i]

                # Ensure prediction is a string for calculations, default to empty if None
                prediction_text = response.prediction if response.prediction is not None else ""

                # Calculate string metrics using the Levenshtein library
                distance = Levenshtein.distance(gt, prediction_text)
                ratio = Levenshtein.ratio(gt, prediction_text)
                # Correctness check: prediction must match GT *and* the operation must have succeeded
                is_correct = 1 if gt == response.prediction and response.success else 0

                # Calculate individual cost
                individual_cost_usd = None
                cost_details_repr = repr(response.cost_details) # Store representation of cost details
                if cost_calculator_for_csv and response.cost_details:
                    try:
                        # Ensure cost calculation handles different CostTypes appropriately
                        individual_cost_usd = cost_calculator_for_csv.calculate_single_cost(response.cost_details)
                    except Exception as cost_calc_e:
                        # Log specific error during cost calculation for this row
                        print(f"Warning: Failed to calculate cost for {model_name} on {image_id} (Index {i}): {cost_calc_e}")
                        individual_cost_usd = None # Ensure cost is None if calculation fails

                # Prepare row data
                row = {
                    'model_name': model_name,
                    'model_version': model_version,
                    'image_id': image_id,
                    'ground_truth': gt,
                    'prediction': response.prediction, # Store the original prediction (might be None)
                    'success': response.success,
                    'error_message': response.error_message,
                    'start_time': response.start_time,
                    'elapsed_time': f"{response.elapsed_time:.4f}" if response.elapsed_time is not None else "", # Format time
                    'cost_usd': f"{individual_cost_usd:.8f}" if individual_cost_usd is not None else "", # Format cost
                    'cost_details_repr': cost_details_repr, # Include the repr
                    'levenshtein_distance': distance,
                    'levenshtein_ratio': f"{ratio:.4f}", # Format ratio
                    'is_correct': is_correct
                }
                writer.writerow(row)

    print("Finished writing detailed results.")

# Catch specific expected errors and general errors during CSV writing
except AttributeError as e:
     print(f"\nError accessing benchmark data for CSV export: {e}")
     print("Ensure the Benchmark object has 'annotations' and 'metadata' (with 'image_filename' keys) attributes.") # Updated error message
     print("Skipping detailed CSV export.")
except ImportError as e:
     # Catch potential import error for string_eval if structure changes
     print(f"\nError importing required module for CSV export: {e}")
     print("Skipping detailed CSV export.")
except Exception as e:
    # Catch any other unexpected error during the process
    print(f"\nAn unexpected error occurred during CSV export: {e}")
    import traceback
    traceback.print_exc()
    print("Skipping detailed CSV export.")
# --- End CSV Export ---

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

# --- Calculate and Print Costs ---
print("\nCalculating Costs...")

# Placeholder for runtime cost per second (e.g., in USD)
runtime_cost_per_second_placeholder = 0.0001 # Example: $0.0001 per second

# Initialize the calculator - stop if this fails
try:
    cost_calculator = ModelCostCalculator.default(runtime_cost_per_second=runtime_cost_per_second_placeholder)
except (ImportError, ValueError) as e:
    print(f"Failed to initialize ModelCostCalculator: {e}. Skipping cost calculations.")
    cost_calculator = None # Set to None so the final print block knows to skip

if cost_calculator: # Only proceed if calculator initialized
    total_benchmark_cost = 0.0
    model_costs_summary = {}

    # Iterate through results for each model
    for model_result in benchmark_results:
        model_info = model_result.model
        model_id_str = f"{model_info.name} ({model_info.version})"
        print(f"  Processing costs for: {model_id_str}")

        # Directly extract cost_details, assuming structure exists
        # This will fail if response is None or lacks cost_details - as requested
        model_costs_list = [
            response.cost_details
            for response in model_result.indexed_results
            if response and isinstance(response.cost_details, ModelCost) # Keep basic check for existence
        ]

        if not model_costs_list:
            print(f"    No cost details found for {model_id_str}.")
            model_costs_summary[model_id_str] = 0.0
            continue

        # Calculate costs - let it fail if calculator encounters issues (e.g., missing pricing)
        individual_costs = cost_calculator.calculate_batch_cost(model_costs_list)

        # Sum costs, filtering out None results from calculator (e.g., for UNKNOWN type)
        total_model_cost = sum(cost for cost in individual_costs if cost is not None)

        model_costs_summary[model_id_str] = total_model_cost
        total_benchmark_cost += total_model_cost
        print(f"    Total estimated cost for {model_id_str}: ${total_model_cost:.6f}")

    # --- Simple Summary ---
    print("\n--- Benchmark Cost Summary ---")
    if model_costs_summary:
        for model_name, cost in model_costs_summary.items():
             # Assume cost is always a float here after the sum()
            print(f"- {model_name}: ${cost:.6f}")
        print(f"\nTotal Estimated Benchmark Cost: ${total_benchmark_cost:.6f}")
        print(f"(Using placeholder runtime cost/sec: ${runtime_cost_per_second_placeholder:.6f})")
    else:
        print("No models had cost details to calculate.")
# --- End Cost Calculation ---

