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
from typing import List


class Benchmark:
    def __init__(self, name, images=[], annotations=[]):
        self.name = name
        self.images = images
        self.annotations = annotations

    def benchmark(
        self,
        models,
        create_autosave=True,
        autosave_dir=None,
        use_autosave=True,
        run_models=True,
        overwrite=False,
    ):
        if autosave_dir is None:
            autosave_dir = f"{self.name}_Autosaves"

        if use_autosave:
            autosave_dir_exists = os.path.exists(autosave_dir)

            if autosave_dir_exists:
                autosaved_model_results_filepaths = os.listdir(autosave_dir)
                autosaved_model_results = {}

                for (
                    autosaved_model_results_filepath
                ) in autosaved_model_results_filepaths:
                    try:
                        autosaved_model_results_path = os.path.join(
                            autosave_dir, autosaved_model_results_filepath
                        )
                        model_result = Benchmark.load(autosaved_model_results_path)

                        assert model_result.benchmark.name == self.name

                        autosaved_model_results.setdefault(model_result.model.name, {})
                        autosaved_model_results[model_result.model.name][
                            model_result.model.version
                        ] = model_result

                        print(
                            "Successfully imported autosaved results:",
                            model_result.model.name,
                            f"(Version: {model_result.model.version})",
                        )
                    except:
                        print(
                            "Failed to import autosave:", autosaved_model_results_path
                        )

            else:
                print(f"Could not find any autosaved model results at", autosave_dir)

        model_results = [BenchmarkModelResult(self, model) for model in models]
        for model_idx, model in enumerate(models):
            model_result = model_results[model_idx]

            print(f"\nTesting Model: {model_result.model.name}")
            print(f"- Version: {model_result.model.version}")

            if use_autosave:
                if model_result.model.name in autosaved_model_results:
                    if (
                        model_result.model.version
                        in autosaved_model_results[model_result.model.name]
                    ):
                        autosaved_model_result = autosaved_model_results[
                            model_result.model.name
                        ][model_result.model.version]

                        # Autosave validation
                        valid_autosave = True

                        ran_images = len(autosaved_model_result.results) + len(
                            autosaved_model_result.failed
                        )
                        if ran_images != len(self.images):
                            print(
                                f"Validation Error: Benchmark image count did not match saved model result. (benchmark: {len(self.images)}, autosave: {ran_images})"
                            )
                            valid_autosave = False

                        if valid_autosave:
                            print(
                                f"Using a autosaved result (Model: {autosaved_model_result.model.name}) (Version: {autosaved_model_result.model.version})"
                            )
                            model_results[model_idx] = autosaved_model_result

                            continue
                        else:
                            print("Not a valid autosaved result. Running benchmark")

            if run_models is False:
                print("Skipping model inference (run_models=False)")
                continue
            for ground_idx, image in tqdm(enumerate(self.images)):
                model_image_result = model.run_for_eval(image)

                model_result.add_result(model_image_result)

            print(f"\nModel: {model_result.model.name} finished testing")
            print(
                f"Successfully Processed %: {len(model_result.results)/(len(model_result.results)+len(model_result.failed)):0.2f}\n"
            )

            if create_autosave and (run_models is True):
                model_result.save(
                    autosave_dir,
                    f"{model_result.model.name}_{model_result.model.version}",
                    overwrite=overwrite,
                )

        return BenchmarkResults(model_results)

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

    def save(self, dir="", path=None, overwrite=False):
        if path is None:
            path = self.id

        file_path = os.path.join(dir, path)

        mode = "wb" if overwrite else "xb"
        try:
            with open(file_path, mode) as outp:
                dill.dump(self, outp)

            saved = BenchmarkModelResult.load(file_path)
            print(saved.__dict__)
            assert len(saved.results) == len(self.results)
            return True
        except BaseException as e:
            print(f"Failed to save model result to {file_path}")
            print("Error:", e)
            traceback.print_exc()
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