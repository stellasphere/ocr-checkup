import traceback
import numpy as np
import time
import hashlib # Import hashlib for config identifier hashing if needed
import json # Import json for config identifier serialization


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

        if success is True and (start_time is not None or elapsed_time is not None):
            raise Exception(
                "Start time and elapsed time should be set by the OCRBaseModel implemenation"
            )

        self.start_time = start_time
        self.elapsed_time = elapsed_time


class OCRModelInfo:
    VALID_TAGS = ["cloud", "lmm", "local"]
    # Added 'local' tag
    VALID_COST_TYPES = ["api", "compute", None] # Added cost types

    def __init__(self, name: str, version: str, tags: list, cost_type: str = None, config_identifier: str = None) -> None: # Added config_identifier
        self.name = name
        self.version = version
        self.tags = tags
        self.cost_type = cost_type # Added cost_type attribute
        self.config_identifier = config_identifier # Added config_identifier attribute

        assert all(tag in self.VALID_TAGS for tag in tags)
        assert cost_type in self.VALID_COST_TYPES


class OCRBaseModel:
    def info(self) -> OCRModelInfo: # Changed to instance method
        # Base info method now constructs the OCRModelInfo using get_config_identifier
        # Subclasses MUST override _get_base_info to provide name, version, etc.
        # This ensures config_identifier is consistently handled.
        base_info = self._get_base_info()
        return OCRModelInfo(
            name=base_info['name'],
            version=base_info['version'],
            tags=base_info['tags'],
            cost_type=base_info.get('cost_type'), # Use .get for optional cost_type
            config_identifier=self.get_config_identifier()
        )

    def _get_base_info(self) -> dict:
        # Subclasses must implement this to return a dict with
        # 'name', 'version', 'tags', and optionally 'cost_type'
        raise NotImplementedError("_get_base_info must be implemented by a subclass")


    def __init__(self, cost_per_second: float = None, **kwargs): # Added **kwargs
        self.cost_per_second = cost_per_second # Store cost_per_second
        self.config_params = kwargs # Store arbitrary configuration parameters
        return None

    def get_config_identifier(self, max_len=50) -> str:
        """Generates a stable, filesystem-safe identifier from config_params."""
        if not self.config_params:
            return "default"

        # Create a sorted, canonical JSON string representation
        try:
            # Sort keys for stability, handle non-string values
            sorted_params = dict(sorted(self.config_params.items()))
            config_str = json.dumps(sorted_params, sort_keys=True, separators=(',', ':'))
        except TypeError:
            # Fallback for non-JSON serializable types (less ideal)
            config_str = str(dict(sorted(self.config_params.items())))


        # Basic sanitization for filesystem compatibility
        # Replace common problematic characters
        safe_str = "".join(c if c.isalnum() or c in ('-', '_', '=', ':') else '_' for c in config_str)
        # Remove potential leading/trailing underscores/hyphens and multiple consecutive underscores
        safe_str = safe_str.strip('_-')
        while '__' in safe_str:
            safe_str = safe_str.replace('__', '_')


        # Optional: Hash if the string is too long (or always hash for consistency)
        # Using SHA-1 for a shorter hash, collision risk is low for typical config counts
        # if len(safe_str) > max_len:
        #     return hashlib.sha1(safe_str.encode()).hexdigest()[:10] # Short hash

        # Return the sanitized string (potentially truncated)
        return safe_str[:max_len].rstrip('_-') # Truncate and clean end


    def test(self):
        try:
            info = self.info() # Call instance method

            img = np.zeros([100, 100, 3], dtype=np.uint8)
            img.fill(255)

            # Use run_for_eval for consistency
            response = self.run_for_eval(img)
            print(f"Test response: {response.prediction}, Cost: {response.cost}, Time: {response.elapsed_time}")
            assert response.success is True

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
        result = None # Initialize result
        try:
            start_time = time.perf_counter()
            result = self.evaluate(image)
            elapsed_time = time.perf_counter() - start_time

            result.elapsed_time = elapsed_time
            result.start_time = start_time

            # Auto-calculate cost if it's a compute model and cost_per_second is set
            # and the evaluate method didn't already provide a cost.
            model_info = self.info()
            if (
                result.cost is None
                and model_info.cost_type == "compute"
                and self.cost_per_second is not None
            ):
                result.cost = elapsed_time * self.cost_per_second

        except Exception as e:
            # Ensure elapsed time is recorded even on failure if possible
            if elapsed_time is None:
                 elapsed_time = time.perf_counter() - start_time

            # Add more detailed error logging
            print(f"--- Exception caught in run_for_eval for model {self.info().name} ---")
            traceback.print_exc()
            print("--- End Exception ---")

            # Create a failure response, preserving time/cost if available
            result = OCRModelResponse(
                 success=False,
                 error_message=str(e), # Use str(e)
                 cost = getattr(result, 'cost', None), # Preserve cost if evaluate partially ran
                 elapsed_time = elapsed_time,
                 start_time = start_time
            )


        # Ensure time is always set, even if evaluate failed early
        if result.elapsed_time is None:
            result.elapsed_time = elapsed_time if elapsed_time is not None else time.perf_counter() - start_time
        if result.start_time is None:
             result.start_time = start_time


        assert isinstance(result, OCRModelResponse), f"Model did not return OCRModelResponse, got {type(result)}"
        assert result.elapsed_time is not None, "Elapsed time not set"
        assert result.start_time is not None, "Start time not set"


        return result
