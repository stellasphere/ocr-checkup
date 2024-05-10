from ocrcheckup import Benchmark

import supervision as sv

from roboflow import Roboflow
rf = Roboflow(api_key="HgKBrG6GzhfssPbg0mEA")
project = rf.workspace("leo-ueno").project("ocr-benchmarking")
version = project.version(4)
dataset = version.download("voc","datasets/industrial-focus-scene")

sv_dataset = sv.DetectionDataset.from_pascal_voc(
    images_directory_path=f"{dataset.location}/test",
    annotations_directory_path=f"{dataset.location}/test",
)

IndustrialSceneBenchmark = Benchmark.from_supervision_dataset("Industrial Focused Scene",sv_dataset)


from ocrcheckup.models import Tesseract

Tesseract().test()

# IndustrialSceneBenchmark.benchmark([TesseractModel()],autosave_dir="testing")