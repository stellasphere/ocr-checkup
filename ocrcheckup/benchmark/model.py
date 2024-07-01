import traceback
import numpy as np
import time


class OCRModelResponse:
    def __init__(
        self,
        prediction: str = None,
        cost: float = None,
        success: bool = True,
        start_time: int = None,
        elapsed_time: int = None,
        error_message: str = "",
    ):
        if prediction is None and success is True:
            raise Exception(
                "There should be a prediction is the response was successfull"
            )

        self.prediction = prediction
        self.cost = cost
        self.success = success
        self.error_message = error_message

        if start_time is not None or elapsed_time is not None:
            raise Exception(
                "Start time and elapsed time should be set by the OCRBaseModel implemenation"
            )

        self.start_time = start_time
        self.elapsed_time = elapsed_time


class OCRModelInfo:
    VALID_TAGS = ["cloud", "lmm"]

    def __init__(self, name: str, version: str, tags: list) -> None:
        self.name = name
        self.version = version
        self.tags = tags

        assert all(tag in self.VALID_TAGS for tag in tags)


class OCRBaseModel:
    def info() -> OCRModelInfo:
        raise NotImplementedError("Info function must be implemented by a subclass")

    def __init__(self):
        return None

    def test(self):
        try:
            info = self.info()

            img = np.zeros([100, 100, 3], dtype=np.uint8)
            img.fill(255)

            self.evaluate(img)
        except:
            print("This model you just initated is not working correctly")
            traceback.print_exc()
            return False

        return True

    def evaluate(self, image) -> OCRModelResponse:
        raise NotImplementedError(
          "Evaluate function must be implemented by a subclass"
        )

    def run_for_eval(self, image):
        try:
            start_time = time.perf_counter()
            result = self.evaluate(image)
            elapsed_time = time.perf_counter() - start_time

            result.elapsed_time = elapsed_time
            result.start_time = start_time
        except BaseException as e:
            result = OCRModelResponse(success=False, error_message=e)

        assert isinstance(result, OCRModelResponse)

        return result
