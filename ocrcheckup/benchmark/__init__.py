from ocrcheckup.benchmark.model import OCRBaseModel, OCRModelResponse
from ocrcheckup.evaluation import StringMetrics
from ocrcheckup.utils import pretty_json

import os
from tqdm import tqdm
import statistics
import dill
import time
import numpy as np
import supervision as sv
import traceback
from typing import List, Union, Optional
import json
from roboflow import Roboflow
import cv2


class Benchmark:
    def __init__(self, name, images=[], annotations=[], metadata=[]):
        self.name = name
        self.images = images
        self.annotations = annotations
        self.metadata = metadata

    def benchmark(
        self,
        models: List[OCRBaseModel],
        autosave_dir: str = None,
        create_autosave: bool = True,
        create_autosave_with_fails: bool = False,
        use_autosave: bool = True,
        overwrite: Union[bool, List[str]] = False,
        run_models: bool = True,
    ):
        """
        Benchmark OCR models against the dataset.
        
        Args:
            models (List[OCRBaseModel]): List of OCR models to benchmark.
            autosave_dir (str, optional): Directory to save benchmark results. Defaults to "{benchmark_name}_Autosaves".
            create_autosave (bool, optional): Whether to save benchmark results. Defaults to True.
            create_autosave_with_fails (bool, optional): If False (default), autosave is only created if all runs for a model succeed (100% success rate). If True, autosave is attempted regardless of success rate.
            use_autosave (bool, optional): Whether to load previously saved results. Defaults to True.
            run_models (bool, optional): Whether to run the models or just load saved results. Defaults to True.
            overwrite (Union[bool, List[str]], optional): Whether to overwrite existing results.
                If True, overwrite all. If False, overwrite none.
                If a list of model names (str), overwrite only those specific models.
                Defaults to False.
            
        Returns:
            BenchmarkResults: Aggregated results of the benchmark.
        """
        if autosave_dir is None:
            autosave_dir = f"{self.name}_Autosaves"

        model_results = [BenchmarkModelResult(self, model) for model in models]
        loaded_results_map = {}

        if use_autosave:
            if not os.path.exists(autosave_dir):
                 print(f"Autosave directory not found, skipping loading: {autosave_dir}")
            else:
                print(f"Checking for autosaved results in: {autosave_dir}")
                for model_idx, model in enumerate(models):
                    model_info = model.info()
                    autosaved_model_results_path = os.path.join(
                        autosave_dir, f"{model_info.name}-{model_info.version}"
                    )

                    if os.path.exists(autosaved_model_results_path):
                        try:
                            autosaved_model_result = BenchmarkModelResult.load(autosaved_model_results_path)
                            print(f"  Found potential autosaved result for {model_info.name} v{model_info.version}")

                            should_skip_loading = False
                            if isinstance(overwrite, bool) and overwrite:
                                print(f"    Overwrite=True, skipping load for {model_info.name}.")
                                should_skip_loading = True
                            elif isinstance(overwrite, list) and model_info.name in overwrite:
                                print(f"    Model {model_info.name} in overwrite list, skipping load.")
                                should_skip_loading = True

                            if not should_skip_loading:
                                valid_autosave = True
                                if not hasattr(autosaved_model_result, 'benchmark') or autosaved_model_result.benchmark.name != self.name:
                                    print(f"    Validation Error: Benchmark name mismatch ('{getattr(autosaved_model_result, 'benchmark', None)}' vs '{self.name}')")
                                    valid_autosave = False
                                elif not hasattr(autosaved_model_result, 'indexed_results') or not isinstance(autosaved_model_result.indexed_results, list):
                                     print(f"    Validation Error: Loaded result missing or invalid 'indexed_results'.")
                                     valid_autosave = False
                                else:
                                     ran_images = len(autosaved_model_result.indexed_results)
                                     expected_images = len(self.images)
                                     if ran_images != expected_images:
                                         print(f"    Validation Error: Image count mismatch (autosave: {ran_images}, expected: {expected_images})")
                                         valid_autosave = False

                                if valid_autosave:
                                    print(f"    Using valid autosaved result for {model_info.name} v{model_info.version}.")
                                    loaded_results_map[model_idx] = autosaved_model_result
                                else:
                                    # Prompt user to delete?
                                    user_input = input(f"    Autosaved result for {model_info.name} is not valid. Will re-run. Delete? (y/n): ")
                                    if user_input.lower() == 'y':
                                        os.remove(autosaved_model_results_path)
                                        print(f"    Deleted autosaved result for {model_info.name}.")
                                        print(f"    Will re-run.")
                                    else:
                                        print(f"    Autosaved result for {model_info.name} is not valid. Will re-run.")
                            # --- End Validation ---
                        except Exception as e:
                            print(f"  Error loading autosaved result for {model_info.name} at {autosaved_model_results_path}: {e}")
                            traceback.print_exc() # Optional: uncomment for more debug info
                    else: # Optional: uncomment for verbose logging
                        print(f"  No autosaved file found for {model_info.name} v{model_info.version} at {autosaved_model_results_path}")
        # --- End loading phase ---

        final_model_results = []
        for model_idx, model in enumerate(models):
            model_result = model_results[model_idx]
            model_info = model_result.model

            print(f"\nProcessing Model: {model_info.name} (Version: {model_info.version})")

            run_this_model = True
            if model_idx in loaded_results_map:
                print(f"  Using loaded result for {model_info.name}.")
                model_result = loaded_results_map[model_idx]
                run_this_model = False
            elif not run_models:
                 print("  Skipping model inference (run_models=False).")
                 run_this_model = False

            if run_this_model:
                print(f"  Running benchmark for {model_info.name}...")
                if len(model_result.indexed_results) != len(self.images):
                     print(f"Warning: Correcting size of indexed_results for {model_info.name}")
                     model_result.indexed_results = [None] * len(self.images)

                for ground_idx, image in tqdm(enumerate(self.images), total=len(self.images), desc=f"  {model_info.name}"):
                    model_image_result = model.run_for_eval(image)
                    model_result.add_result(ground_idx, model_image_result)

                print(f"  Model {model_info.name} finished testing.")
                total_processed = sum(1 for res in model_result.indexed_results if res is not None)
                total_successful = sum(1 for res in model_result.indexed_results if res is not None and res.success)
                success_percentage = (float(total_successful) / total_processed * 100.0) if total_processed > 0 else 0.0
                print(f"  Successfully Processed: {success_percentage:.2f}% ({total_successful}/{total_processed})")

                if create_autosave:
                    can_autosave = True
                    if not create_autosave_with_fails and success_percentage < 100.0:
                        print(f"  Skipping autosave for {model_info.name}: Success rate ({success_percentage:.2f}%) is less than 100% and create_autosave_with_fails is False.")
                        can_autosave = False

                    if can_autosave:
                        should_overwrite_save = False
                        if isinstance(overwrite, bool):
                            should_overwrite_save = overwrite
                        elif isinstance(overwrite, list) and model_info.name in overwrite:
                            should_overwrite_save = True

                        print(f"  Attempting to save result for {model_info.name} (overwrite={should_overwrite_save})...")
                        saved_path = model_result.save(
                            autosave_dir,
                            path=None,
                            overwrite=should_overwrite_save,
                        )
                        if saved_path:
                            print(f"  Result saved to: {saved_path}")
                        else:
                            print(f"  Failed to save result for {model_info.name} (maybe file existed and overwrite was False).")
            else:
                 if create_autosave and model_idx in loaded_results_map:
                     should_force_save = False
                     if isinstance(overwrite, bool) and overwrite:
                         should_force_save = True
                     elif isinstance(overwrite, list) and model_info.name in overwrite:
                         should_force_save = True

                     if should_force_save:
                         print(f"  Force-saving loaded result for {model_info.name} due to overwrite flag...")
                         saved_path = model_result.save(
                             autosave_dir,
                             path=None,
                             overwrite=True,
                         )
                         if saved_path:
                             print(f"  Result saved to: {saved_path}")
                     else:
                          print(f"  Skipping save for loaded result {model_info.name} (overwrite flag not set).")

            final_model_results.append(model_result)

        return BenchmarkResults(final_model_results)

    def save(self, path=None):
        if path is None:
            path = self.name

        with open(path, "wb") as outp:
            dill.dump(self, outp)

        return path

    def load(path):
        with open(path, "rb") as inp:
            data = dill.load(inp)
        return data

    def showcase(self, max_count=12, size=(12, 12), grid_size=(3, 4)):
        assert max_count <= grid_size[0] * grid_size[1]

        images = self.images[:max_count]
        titles = self.annotations[:max_count]

        sv.plot_images_grid(images, grid_size=grid_size, titles=titles, size=size)

    @classmethod
    def from_supervision_dataset(cls, name, sv_dataset):
        images = []
        annotations = []

        iterable = enumerate(tqdm(sv_dataset))
        for idx, (_, image, annotation) in iterable:
            images.append(image)

            ground_truth = sv_dataset.classes[annotation[0].class_id[0]]
            annotations.append(ground_truth)

        return cls(name, images, annotations)

    @classmethod
    def from_roboflow_dataset(cls, name: str, api_key: str, workspace: str, project: str, version: int, variant: str = "jsonl"):
        """
        Creates a Benchmark instance from a Roboflow dataset.

        Downloads the specified dataset version using the Roboflow API,
        then loads images and annotations from the 'test' split.
        Assumes the 'jsonl' variant where annotations are in 'annotations.jsonl'
        and contain 'image' and 'suffix' keys.

        Args:
            name (str): The name for this benchmark instance.
            api_key (str): Your Roboflow API key.
            workspace (str): The Roboflow workspace ID or URL name.
            project (str): The Roboflow project ID or URL name.
            version (int): The dataset version number.
            variant (str, optional): The dataset variant to download. Defaults to "jsonl".

        Returns:
            Benchmark: A new Benchmark instance populated with the test data.
        """
        print(f"Downloading Roboflow dataset: {workspace}/{project}/{version}")
        rf = Roboflow(api_key=api_key)
        project_obj = rf.workspace(workspace).project(project)
        version_obj = project_obj.version(version)
        dataset = version_obj.download("jsonl", f"datasets/{name}")

        test_dir = os.path.join(dataset.location, "test")
        annotations_path = os.path.join(test_dir, "annotations.jsonl")

        if not os.path.exists(annotations_path):
            raise FileNotFoundError(f"Annotations file not found at expected location: {annotations_path}")

        images = []
        annotations = []
        metadata_list = []

        print(f"Loading data from: {annotations_path}")
        with open(annotations_path, 'r') as f:
            for line in tqdm(f, desc=f"Processing {name} test set"):
                try:
                    data = json.loads(line.strip())
                    image_filename = data.get("image")
                    ground_truth = data.get("suffix")

                    if not image_filename or ground_truth is None:
                        print(f"Warning: Skipping line due to missing 'image' or 'suffix': {line.strip()}")
                        raise Exception(f"Missing 'image' or 'suffix': {line.strip()}")

                    image_path = os.path.join(test_dir, image_filename)
                    if not os.path.exists(image_path):
                        print(f"Warning: Image file not found, skipping: {image_path}")
                        raise Exception(f"Image file not found: {image_path}")

                    image = cv2.imread(image_path)
                    if image is None:
                        print(f"Warning: Failed to load image, skipping: {image_path}")
                        raise Exception(f"Failed to load image: {image_path}")

                    images.append(image)
                    annotations.append(ground_truth)

                    metadata_item = {}
                    if image_filename:
                         metadata_item['image_filename'] = image_filename

                    for key, value in data.items():
                        if key not in ["image", "suffix"]:
                             metadata_item[key] = value
                    metadata_list.append(metadata_item)

                except json.JSONDecodeError:
                    print(f"Warning: Skipping invalid JSON line: {line.strip()}")
                except Exception as e:
                    print(f"Warning: Error processing line '{line.strip()}': {e}")

        print(f"Loaded {len(images)} images, annotations, and metadata entries for benchmark '{name}'.")
        return cls(name, images, annotations, metadata_list)


class BenchmarkModelResult:
    def __init__(
        self,
        benchmark: Benchmark,
        model: OCRBaseModel,
        created: Optional[float] = None,
        indexed_results: Optional[List[Optional[OCRModelResponse]]] = None
    ):
        self.benchmark = benchmark
        self.model = model.info()
        self.created = created if created is not None else time.time()

        num_images = len(benchmark.images) if benchmark and hasattr(benchmark, 'images') else 0
        self.indexed_results: List[Optional[OCRModelResponse]] = indexed_results if indexed_results is not None else [None] * num_images

        self.id = f"{self.benchmark.name}_{self.model.name}-{self.model.version}_{self.created}"

    def add_result(self, image_index: int, result: OCRModelResponse, verbose=False):
        if verbose:
            print(f"Result for index {image_index}:", pretty_json(result.__dict__))

        if 0 <= image_index < len(self.indexed_results):
            self.indexed_results[image_index] = result
        else:
            print(f"Warning: image_index {image_index} out of bounds for indexed_results (size: {len(self.indexed_results)}). Result not added.")

    def save(self, dir="", path=None, overwrite=False, create_dir=True):
        if path is None:
            path = f"{self.model.name}-{self.model.version}"

        file_path = os.path.join(dir, path)

        if create_dir:
            file_dir = os.path.dirname(file_path)
            if file_dir:
                os.makedirs(file_dir, exist_ok=True)

        mode = "wb" if overwrite else "xb"
        try:
            with open(file_path, mode) as outp:
                dill.dump(self, outp)

            try:
                saved = BenchmarkModelResult.load(file_path)
                assert hasattr(saved, 'indexed_results'), "Loaded object missing 'indexed_results'"
                assert isinstance(saved.indexed_results, list), "'indexed_results' is not a list"
                assert len(saved.indexed_results) == len(self.indexed_results), \
                    f"Length mismatch: loaded {len(saved.indexed_results)}, expected {len(self.indexed_results)}"
                return file_path
            except Exception as load_err:
                 print(f"Warning: Saved file {file_path}, but failed validation check on reload: {load_err}")
                 return None

        except FileExistsError:
             return None
        except Exception as e:
            print(f"Failed to save model result to {file_path}")
            print(f"Error: {e}")
            return None

    @staticmethod
    def load(path):
        with open(path, "rb") as inp:
            data = dill.load(inp)
        if not isinstance(data, BenchmarkModelResult):
            raise TypeError(f"Loaded object is not of type BenchmarkModelResult: {type(data)}")
        if not hasattr(data, 'indexed_results'):
             raise AttributeError(f"Loaded BenchmarkModelResult is missing 'indexed_results' attribute from path: {path}")
        return data

    def showcase(self, max_count=12, size=(12, 12), grid_size=(3, 4)):
        if max_count > len(self.indexed_results):
             max_count = len(self.indexed_results)
             print(f"Warning: max_count reduced to {max_count} (number of results)")

        if max_count == 0:
             print("No results to showcase.")
             return

        assert max_count <= grid_size[0] * grid_size[1], \
            f"Grid size {grid_size} is too small for max_count {max_count}"

        images = self.benchmark.images[:max_count]

        titles = []
        for i in range(max_count):
             result = self.indexed_results[i]
             if result and result.success:
                 titles.append(result.prediction)
             elif result and not result.success:
                  titles.append(f"FAILED: {result.error_message[:30]}...")
             else:
                  titles.append("MISSING")

        sv.plot_images_grid(images, grid_size=grid_size, titles=titles, size=size)

    def failed_percent(self):
        if not self.indexed_results:
            return 0.0

        num_images = len(self.indexed_results)
        num_failed = sum(1 for res in self.indexed_results if res is not None and not res.success)

        num_processed = sum(1 for res in self.indexed_results if res is not None)

        if num_processed == 0:
             return 0.0

        return (num_failed / num_processed) * 100 if num_processed > 0 else 0.0

class BenchmarkResults(list):
    def __init__(self, results: List[BenchmarkModelResult]):
        super().__init__(results)