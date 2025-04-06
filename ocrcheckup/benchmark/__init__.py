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
from typing import List, Union


class Benchmark:
    def __init__(self, name, images=[], annotations=[]):
        self.name = name
        self.images = images
        self.annotations = annotations

    def benchmark(
        self,
        models: List[OCRBaseModel],
        autosave_dir: str = None,
        create_autosave: bool = True,
        use_autosave: bool = True,
        overwrite: Union[bool, List[str]] = False,
        run_models: bool = True,
    ):
        """
        Benchmark OCR models against the dataset.
        
        Args:
            models (List[OCRBaseModel]): List of OCR models to benchmark.
            create_autosave (bool, optional): Whether to save benchmark results. Defaults to True.
            autosave_dir (str, optional): Directory to save benchmark results. Defaults to "{benchmark_name}_Autosaves".
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
                        autosave_dir, f"{model_info.name}_{model_info.version}"
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
                                if autosaved_model_result.benchmark.name != self.name:
                                    print(f"    Validation Error: Benchmark name mismatch ('{autosaved_model_result.benchmark.name}' vs '{self.name}')")
                                    valid_autosave = False
                                else:
                                    ran_images = len(autosaved_model_result.results) + len(autosaved_model_result.failed)
                                    if ran_images != len(self.images):
                                        print(f"    Validation Error: Image count mismatch (benchmark: {len(self.images)}, autosave: {ran_images})")
                                        valid_autosave = False

                                if valid_autosave:
                                    print(f"    Using valid autosaved result for {model_info.name} v{model_info.version}.")
                                    loaded_results_map[model_idx] = autosaved_model_result
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
                model_result.results = np.array([])
                model_result.start_times = np.array([])
                model_result.elapsed_times = np.array([])
                model_result.failed = np.array([])

                for ground_idx, image in tqdm(enumerate(self.images), total=len(self.images), desc=f"  {model_info.name}"):
                    model_image_result = model.run_for_eval(image)
                    model_result.add_result(model_image_result)

                print(f"  Model {model_info.name} finished testing.")
                total_processed = len(model_result.results) + len(model_result.failed)
                success_percentage = (len(model_result.results) / total_processed * 100) if total_processed > 0 else 0
                print(f"  Successfully Processed: {success_percentage:.2f}%")

                if create_autosave:
                    should_overwrite_save = False
                    if isinstance(overwrite, bool):
                        should_overwrite_save = overwrite
                    elif isinstance(overwrite, list) and model_info.name in overwrite:
                         should_overwrite_save = True

                    print(f"  Attempting to save result for {model_info.name} (overwrite={should_overwrite_save})...")
                    saved_path = model_result.save(
                        autosave_dir,
                        f"{model_info.name}_{model_info.version}",
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
                             f"{model_info.name}_{model_info.version}",
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


class BenchmarkModelResult:
    def __init__(
        self,
        benchmark: Benchmark,
        model: OCRBaseModel,
        start_times=[],
        elapsed_times=[],
        results=[],
        created=None,
    ):

        self.benchmark = benchmark
        self.model = model.info()
        self.start_times = np.array(start_times)
        self.elapsed_times = np.array(elapsed_times)
        self.results = np.array(results)
        self.created = created if created is not None else time.time()
        self.failed = np.array([])
        self.id = f"{self.benchmark.name}_{self.model.name}-{self.model.version}_{self.created}"

    def add_result(self, result: OCRModelResponse, verbose=False):
        if verbose:
            print("Result:", pretty_json(result.__dict__))

        if result.success is False:
            if verbose:
                print("Failed, adding to failed list")
            self.failed = np.append(self.failed, result)
            return

        if verbose:
            print("Success, adding to results list")
        self.start_times = np.append(self.start_times, result.start_time)
        self.elapsed_times = np.append(self.elapsed_times, result.elapsed_time)
        self.results = np.append(self.results, result.prediction)

    def save(self, dir="", path=None, overwrite=False, create_dir=True):
        if path is None:
            path = self.id

        file_path = os.path.join(dir, path)

        if create_dir:
            # Get the directory containing the target file
            file_dir = os.path.dirname(file_path)
            # Create the full directory path if it doesn't exist
            if file_dir: # Ensure file_dir is not empty if file_path is just a filename
                os.makedirs(file_dir, exist_ok=True)

        mode = "wb" if overwrite else "xb"
        try:
            with open(file_path, mode) as outp:
                dill.dump(self, outp)

            saved = BenchmarkModelResult.load(file_path)
            print(saved.__dict__)
            assert len(saved.results) == len(self.results)
            return True
        except FileExistsError: # More specific exception handling
             print(f"Failed to save result for {path} because file exists and overwrite is False.")
             return False
        except Exception as e: # Catch other potential exceptions
            print(f"Failed to save model result to {file_path}")
            print(f"Error: {e}")
            # traceback.print_exc() # Optional: Keep for detailed debugging if needed
            return False

    def load(path):
        with open(path, "rb") as inp:
            data = dill.load(inp)
        return data

    def showcase(self, max_count=12, size=(12, 12), grid_size=(3, 4)):
        assert max_count <= grid_size[0] * grid_size[1]

        images = self.benchmark.images[:max_count]
        titles = self.results.tolist()[:max_count]

        sv.plot_images_grid(images, grid_size=grid_size, titles=titles, size=size)

    def failed_percent(self):
        failed = self.failed / (self.failed + len(self.results))
        return failed

class BenchmarkResults(list):
    def __init__(self, results: List[BenchmarkModelResult]):
        super().__init__(results)