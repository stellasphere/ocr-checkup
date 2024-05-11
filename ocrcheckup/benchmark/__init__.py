from ocrcheckup.models import OCRBaseModel
from ocrcheckup.evaluation import StringMetrics

import os
from tqdm import tqdm
import statistics
import dill
import time
import numpy as np
import supervision as sv

class Benchmark():
  def __init__(self, name, images=[], annotations=[]):
    self.name = name
    self.images = images
    self.annotations = annotations

  def benchmark(self, models, autosave=True, autosave_dir=None, use_autosave=True):
    if autosave_dir is None:
        autosave_dir = f"{self.name}_Autosaves"

    if use_autosave:
      autosave_dir_exists = os.path.exists(autosave_dir)

      if autosave_dir_exists:
        autosaved_model_results_filepaths = os.listdir(autosave_dir)
        autosaved_model_results = {}

        for autosaved_model_results_filepath in autosaved_model_results_filepaths:
          try:
            autosaved_model_results_path = os.path.join(autosave_dir,autosaved_model_results_filepath)
            model_result = Benchmark.load(autosaved_model_results_path)

            assert model_result.benchmark.name == self.name

            autosaved_model_results.setdefault(model_result.model.name,{})
            autosaved_model_results[model_result.model.name][model_result.model.version] = model_result

            print("Successfully imported autosaved results:",model_result.model.name,f"(Version: {model_result.model.version})")
          except:
            print("Failed to import autosave:",autosaved_model_results_path)


      else:
        print(f"Could not find any autosaved model results at",autosave_dir)


    model_results = [BenchmarkModelResult(self,model) for model in models]
    for model_idx, model in enumerate(models):
      model_result = model_results[model_idx]
      
      print(f"\nTesting Model: {model_result.model.name}")
      print(f"- Version: {model_result.model.version}")

      if use_autosave:
        if model_result.model.name in autosaved_model_results:
          if model_result.model.version in autosaved_model_results[model_result.model.name]:
            autosaved_model_result = autosaved_model_results[model_result.model.name][model_result.model.version]

            # Autosave validation
            valid_autosave = True
            if len(autosaved_model_result.results) != len(self.images):
              print("Validation Error: Benchmark image count did not match saved model result")
              valid_autosave = False

            if valid_autosave:
              print(f"Using a autosaved result (Model: {model_result.model.name}) (Version: {model_result.model.version})")
              model_results[model_idx] = autosaved_model_result

              continue
            else:
              print("Not a valid autosaved result. Running benchmark")

      for ground_idx, image in tqdm(enumerate(self.images)):
        model_image_result = model.run_for_eval(image)

        model_result.add_result(model_image_result)

      if autosave:
        model_result.save(autosave_dir,f"{model_result.model.name}_{model_result.model.version}")


    return model_results

  def save(self,path=None):
    if path is None: path = self.name;

    with open(path, 'wb') as outp:
      dill.dump(self,outp)

    return path

  def load(path):
    with open(path, 'rb') as inp:
      data = dill.load(inp)
    return data

  def showcase(self, max_count=12, size=(12,12)):
    images = self.images[:max_count]
    titles = self.annotations[:max_count]
    
    sv.plot_images_grid(images, grid_size=(3, 4), titles=titles, size=size)

  @classmethod
  def from_supervision_dataset(cls,name,sv_dataset):
    images = []
    annotations = []

    iterable = enumerate(tqdm(sv_dataset))
    for idx, (_, image, annotation) in iterable:
      images.append(image)

      ground_truth = sv_dataset.classes[annotation[0].class_id[0]]
      annotations.append(ground_truth)

    return cls(name,images,annotations)

class BenchmarkModelResult():
  def __init__(self,benchmark:Benchmark,model:OCRBaseModel,start_times=[],elapsed_times=[],results=[],created=None):

    self.benchmark = benchmark
    self.model = model
    self.start_times = np.array(start_times)
    self.elapsed_times = np.array(elapsed_times)
    self.results = np.array(results)
    self.created = created if created is not None else time.time()
    self.id = f"{self.benchmark.name}_{self.model.name}-{self.model.version}_{self.created}"

  def add_result(self, result_object):
    self.start_times = np.append(self.start_times, result_object["start_time"])
    self.elapsed_times = np.append(self.elapsed_times, result_object["elapsed_time"])
    self.results = np.append(self.results, result_object["result"])


  def save(self,dir="",path=None):
    if path is None: path = self.id;

    file_path = os.path.join(dir,path)

    with open(file_path, 'wb') as outp:
      dill.dump(self,outp)

  def load(path):
    with open(path, 'rb') as inp:
      data = dill.load(inp)
    return data
  
  def showcase(self, max_count=12, size=(12,12)):
    images = self.benchmark.images[:max_count]
    titles = self.results.tolist()[:max_count]
    
    sv.plot_images_grid(images, grid_size=(3, 4), titles=titles, size=size)


class BenchmarkModelMetrics():
  def __init__(self,benchmark_result:BenchmarkModelResult,result_eval_methods=StringMetrics.EVALUATION_METHODS):
    self.results = benchmark_result.results
    self.benchmark = benchmark_result.benchmark
    self.model = benchmark_result.model
    self.ground_truths = self.benchmark.annotations

    # Result Scores
    self.scores = {}
    for idx, result in enumerate(self.results):
      ground_truth = self.ground_truths[idx]

      result_evaluated = StringMetrics(result,ground_truth).evaluate(result_eval_methods)
      for method_idx, method_score in enumerate(result_evaluated):
        self.scores.setdefault(result_eval_methods[method_idx],[])
        self.scores[result_eval_methods[method_idx]].append(method_score)

  def averages(self):
    averages = {}
    for method, score_array in self.scores.items():
      averages[method] = statistics.mean(score_array)

    return averages
  
