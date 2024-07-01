from ocrcheckup import Benchmark
import ocrcheckup

import supervision as sv
import os

from roboflow import Roboflow

import ocrcheckup.evaluation
import ocrcheckup.models
import ocrcheckup.utils
from ocrcheckup.models import DocTR_RFHosted, OpenAI_GPT4o

rf = Roboflow(api_key=os.environ["ROBOFLOW_API_KEY"])
project = rf.workspace("leo-ueno").project("ocr-benchmarking")
version = project.version(4)
dataset = version.download("voc", "datasets/industrial-focus-scene")

sv_dataset = sv.DetectionDataset.from_pascal_voc(
    images_directory_path=f"{dataset.location}/test",
    annotations_directory_path=f"{dataset.location}/test",
)

IndustrialSceneBenchmark = Benchmark.from_supervision_dataset(
    "Industrial Focused Scene", sv_dataset
)




from ocrcheckup.benchmark.model import OCRBaseModel, OCRModelResponse, OCRModelInfo


class TestModel(OCRBaseModel):
    def info(self=None):
        return OCRModelInfo(
            name="Test Model", version="v1", tags=[]  # cloud, lmm, local
        )

    def __init__(self):
        super().__init__()

    def evaluate(self, image):

        return OCRModelResponse(prediction="text", cost=0.00)


models = [
    DocTR_RFHosted(api_key=os.environ["ROBOFLOW_API_KEY"]),
    OpenAI_GPT4o(api_key=os.environ["OPENAI_API_KEY"]),
    TestModel(),
]

benchmark_results = IndustrialSceneBenchmark.benchmark(
    models,
    autosave_dir="testing",
    create_autosave=True,
    run_models=True,
    overwrite=True,
)
print("Benchmark Results:", type(benchmark_results))

string_metrics = ocrcheckup.evaluation.StringMetrics.from_benchmark_model_results(
    benchmark_results
)
print(ocrcheckup.utils.pretty_json(string_metrics))

speed_metrics = ocrcheckup.evaluation.SpeedMetrics.from_benchmark_model_results(
    benchmark_results
)
print(ocrcheckup.utils.pretty_json(speed_metrics))
