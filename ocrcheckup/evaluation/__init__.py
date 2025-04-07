import Levenshtein
import statistics
from typing import List, Any, Optional
from ..benchmark.model import OCRModelResponse


class ValuesSummarization:
    def __init__(self, values: List[Any]):
        self.values = values

    SUMMARIZATION_METHODS = ["average", "median"]

    def summarize(self, methods=SUMMARIZATION_METHODS):
        return tuple(getattr(self, method)() for method in methods)

    def average(self):
        return statistics.mean(self.values)

    def median(self):
        return statistics.median(self.values)


class StringMetrics:
    def __init__(
        self,
        a,
        b,
        capitalization_sensitive=True,
        linebreak_sensitive=True,
        stripping=True,
    ):

        self.a = a
        self.b = b
        self.capitalization_sensitive = capitalization_sensitive
        self.linebreak_sensitive = linebreak_sensitive
        self.stripping = stripping

        if not self.capitalization_sensitive:
            self.a, self.b = self.lowercase(self.a, self.b)

        if not linebreak_sensitive:
            self.a, self.b = self.remove_linebreaks(self.a, self.b)

        if stripping:
            self.a = self.a.strip()
            self.b = self.b.strip()

    @staticmethod
    def lowercase(*args):
        return tuple(string.lower() for string in args)

    @staticmethod
    def remove_linebreaks(*args):
        return tuple(" ".join(string.splitlines()) for string in args)

    EVALUATION_METHODS = ["levenshtein_ratio", "correct"]

    def levenshtein_ratio(self):
        ratio = Levenshtein.ratio(self.a, self.b)
        return ratio

    def correct(self):
        correct = 1 if self.a == self.b else 0
        return correct

    def evaluate(self, methods=EVALUATION_METHODS):
        return tuple(getattr(self, method)() for method in methods)

    @staticmethod
    def from_benchmark_model_result(
        benchmark_result,
        eval_methods=EVALUATION_METHODS,
        summarization_methods=ValuesSummarization.SUMMARIZATION_METHODS,
    ):
        """Calculates string metrics for a single BenchmarkModelResult using indexed_results."""
        eval_results = {}
        # Initialize score lists within eval_results
        for method in eval_methods:
            eval_results.setdefault(method, {})
            eval_results[method]["scores"] = []

        # Check if indexed_results exists and is iterable
        if not hasattr(benchmark_result, 'indexed_results') or not isinstance(benchmark_result.indexed_results, list):
             print(f"Warning: BenchmarkModelResult for model {getattr(benchmark_result.model, 'name', '?')} missing or invalid indexed_results. Skipping string metric calculation.")
             # Return empty/default structure to avoid downstream errors
             for method in eval_methods:
                 for sm_method in summarization_methods:
                      eval_results[method][sm_method] = None # Or 0.0 depending on desired handling
             return eval_results

        # Iterate through indexed_results
        for idx, response in enumerate(benchmark_result.indexed_results):
            # Check if response exists and was successful
            if response is not None and isinstance(response, OCRModelResponse) and response.success:
                try:
                    ground_truth = benchmark_result.benchmark.annotations[idx]
                    prediction = response.prediction

                    if prediction is None: # Should not happen if success is True, but check
                        print(f"Warning: Successful response at index {idx} for model {benchmark_result.model.name} has None prediction. Skipping.")
                        continue

                    # Calculate metrics for this prediction/ground_truth pair
                    result_evaluated = StringMetrics(prediction, ground_truth).evaluate(eval_methods)

                    for method_idx, method_score in enumerate(result_evaluated):
                        eval_method = eval_methods[method_idx]
                        eval_results[eval_method]["scores"].append(method_score)

                except IndexError:
                     print(f"Warning: Annotation index {idx} out of bounds. Skipping string metric calculation for this item.")
                except Exception as e:
                     print(f"Warning: Error processing string metrics for index {idx}, model {benchmark_result.model.name}: {e}")
            # else: response was None, not OCRModelResponse, or not successful - skip metric calculation

        # Summarize collected scores
        for eval_method, eval_data in eval_results.items():
            scores = eval_data.get("scores", [])
            if scores: # Only summarize if there are scores
                summarized = ValuesSummarization(scores).summarize(summarization_methods)
                for idx, summarization_method in enumerate(summarization_methods):
                    eval_results[eval_method][summarization_method] = summarized[idx]
            else:
                 # Handle case with no successful results for this metric
                 for summarization_method in summarization_methods:
                     eval_results[eval_method][summarization_method] = None # Or 0.0, depending on desired handling

        return eval_results

    @staticmethod
    def from_benchmark_model_results(
        benchmark_results,
        eval_methods=EVALUATION_METHODS,
        summarization_methods=ValuesSummarization.SUMMARIZATION_METHODS,
        summarize=True,
        handle_empty_results='error'
    ):
        """
        Calculates string evaluation metrics across multiple benchmark model results.
        Uses the static from_benchmark_model_result method internally.
        (Handles aggregation and error/ignore/zero logic for summaries)
        """
        allowed_handling = ['error', 'ignore', 'zero']
        if handle_empty_results not in allowed_handling:
            raise ValueError(f"handle_empty_results must be one of {allowed_handling}, got '{handle_empty_results}'")

        # --- Use the updated from_benchmark_model_result --- #
        eval_results_per_model = []
        valid_benchmark_results = [] # Keep track of results we could process
        for idx, benchmark_result in enumerate(benchmark_results):
             # Basic check if it looks like a valid BenchmarkModelResult
             if hasattr(benchmark_result, 'model') and hasattr(benchmark_result, 'indexed_results'):
                 model_eval_results = StringMetrics.from_benchmark_model_result(
                     benchmark_result, eval_methods, summarization_methods
                 )
                 eval_results_per_model.append(model_eval_results)
                 valid_benchmark_results.append(benchmark_result)
             else:
                  print(f"Warning: Skipping benchmark result at index {idx} due to missing attributes (model or indexed_results).")

        if not summarize:
            return eval_results_per_model # Return list of results per model

        # --- Aggregation logic (remains largely the same, but uses valid_benchmark_results) ---
        summarized_results = {}
        for eval_method in eval_methods:
            summarized_results[eval_method] = {}
            for summarization_method in summarization_methods:
                summarized_results[eval_method][summarization_method] = {}
                # Iterate through the results we actually processed
                for idx, eval_result in enumerate(eval_results_per_model):
                    # Get model name from the corresponding valid benchmark result
                    model_name = valid_benchmark_results[idx].model.name

                    # Check if the metric and summary method exist AND the value is not None
                    metric_summary_value = eval_result.get(eval_method, {}).get(summarization_method)

                    if metric_summary_value is not None:
                        summarized_results[eval_method][summarization_method][model_name] = metric_summary_value
                    else:
                        # Handle missing summary based on handle_empty_results
                        if handle_empty_results == 'error':
                            raise KeyError(
                                f"Metric summary '{eval_method}' -> '{summarization_method}' not found or was None for model '{model_name}'. "
                                f"This often happens with 0 successful results. "
                                f"Set handle_empty_results='ignore' or handle_empty_results='zero' to change behavior."
                            )
                        elif handle_empty_results == 'ignore':
                            pass # Skip adding the entry for this model/metric/summary
                        elif handle_empty_results == 'zero':
                            summarized_results[eval_method][summarization_method][model_name] = 0.0

        return summarized_results


class SpeedMetrics:
    EVALUATION_METHODS = ["elapsed_time"]

    @staticmethod
    def from_benchmark_model_result(
        benchmark_result,
        eval_methods=EVALUATION_METHODS,
        summarization_methods=ValuesSummarization.SUMMARIZATION_METHODS,
    ):
        """Calculates speed metrics for a single BenchmarkModelResult using indexed_results."""
        eval_results = {}
        all_elapsed_times = [] # Collect all valid elapsed times

        # Check for indexed_results
        if not hasattr(benchmark_result, 'indexed_results') or not isinstance(benchmark_result.indexed_results, list):
             print(f"Warning: BenchmarkModelResult for model {getattr(benchmark_result.model, 'name', '?')} missing or invalid indexed_results. Skipping speed metric calculation.")
             # Return empty/default structure
             for method in eval_methods:
                 eval_results.setdefault(method, {})
                 eval_results[method]["scores"] = []
                 for sm_method in summarization_methods:
                     eval_results[method][sm_method] = None
             return eval_results

        # Iterate through indexed_results
        for idx, response in enumerate(benchmark_result.indexed_results):
             # Check if response exists, is OCRModelResponse, and has elapsed_time
             if response is not None and isinstance(response, OCRModelResponse) and hasattr(response, 'elapsed_time') and response.elapsed_time is not None:
                 all_elapsed_times.append(response.elapsed_time)
             # else: Skip entries without valid elapsed time

        # Populate eval_results structure (currently only 'elapsed_time')
        eval_method = eval_methods[0] # Assuming only 'elapsed_time'
        eval_results.setdefault(eval_method, {})
        eval_results[eval_method]["scores"] = all_elapsed_times

        # Summarize collected scores
        if all_elapsed_times:
            summarized = ValuesSummarization(all_elapsed_times).summarize(summarization_methods)
            for idx, summarization_method in enumerate(summarization_methods):
                eval_results[eval_method][summarization_method] = summarized[idx]
        else:
            # Handle case with no valid elapsed times
            for summarization_method in summarization_methods:
                eval_results[eval_method][summarization_method] = None # Or 0.0?

        # Ensure structure consistency for other potential future methods
        for method in eval_methods:
             if method not in eval_results:
                  eval_results[method] = {sm: None for sm in summarization_methods}
                  eval_results[method]["scores"] = []

        return eval_results

    @staticmethod
    def from_benchmark_model_results(
        benchmark_results,
        eval_methods=EVALUATION_METHODS,
        summarization_methods=ValuesSummarization.SUMMARIZATION_METHODS,
        summarize=True,
        handle_empty_results='error'
    ):
        """
        Calculates speed evaluation metrics across multiple benchmark model results.
        Uses the static from_benchmark_model_result method internally.
        """
        allowed_handling = ['error', 'ignore', 'zero']
        if handle_empty_results not in allowed_handling:
            raise ValueError(f"handle_empty_results must be one of {allowed_handling}, got '{handle_empty_results}'")

        # --- Use the updated from_benchmark_model_result --- #
        eval_results_per_model = []
        valid_benchmark_results = []
        for idx, benchmark_result in enumerate(benchmark_results):
             if hasattr(benchmark_result, 'model') and hasattr(benchmark_result, 'indexed_results'):
                 model_eval_results = SpeedMetrics.from_benchmark_model_result(
                     benchmark_result, eval_methods, summarization_methods
                 )
                 eval_results_per_model.append(model_eval_results)
                 valid_benchmark_results.append(benchmark_result)
             else:
                 print(f"Warning: Skipping benchmark result at index {idx} due to missing attributes (model or indexed_results).")


        if not summarize:
            return eval_results_per_model

        # --- Aggregation logic (similar to StringMetrics) ---
        summarized_results = {}
        for eval_method in eval_methods:
            summarized_results[eval_method] = {}
            for summarization_method in summarization_methods:
                summarized_results[eval_method][summarization_method] = {}
                for idx, eval_result in enumerate(eval_results_per_model):
                    model_name = valid_benchmark_results[idx].model.name
                    metric_summary_value = eval_result.get(eval_method, {}).get(summarization_method)

                    if metric_summary_value is not None:
                        summarized_results[eval_method][summarization_method][model_name] = metric_summary_value
                    else:
                        if handle_empty_results == 'error':
                            raise KeyError(
                                f"Metric summary '{eval_method}' -> '{summarization_method}' not found or was None for model '{model_name}'. "
                                f"Often happens with 0 results. Set handle_empty_results='ignore'/'zero'."
                            )
                        elif handle_empty_results == 'ignore':
                            pass
                        elif handle_empty_results == 'zero':
                            summarized_results[eval_method][summarization_method][model_name] = 0.0

        return summarized_results


class CostMetrics:
    EVALUATION_METHODS = ["cost"]

    @staticmethod
    def from_benchmark_model_result(
        benchmark_result,
        eval_methods=EVALUATION_METHODS,
        summarization_methods=ValuesSummarization.SUMMARIZATION_METHODS,
    ):
        """Calculates cost metrics for a single BenchmarkModelResult using indexed_results."""
        eval_results = {}
        all_costs = [] # Collect all valid costs

        # Check for indexed_results
        if not hasattr(benchmark_result, 'indexed_results') or not isinstance(benchmark_result.indexed_results, list):
             print(f"Warning: BenchmarkModelResult for model {getattr(benchmark_result.model, 'name', '?')} missing or invalid indexed_results. Skipping cost metric calculation.")
             # Return empty/default structure
             for method in eval_methods:
                 eval_results.setdefault(method, {})
                 eval_results[method]["scores"] = []
                 for sm_method in summarization_methods:
                     eval_results[method][sm_method] = None
             return eval_results

        # Iterate through indexed_results
        for idx, response in enumerate(benchmark_result.indexed_results):
             # Check if response exists, is OCRModelResponse, has cost, and cost is not None
             if response is not None and isinstance(response, OCRModelResponse) and hasattr(response, 'cost') and response.cost is not None:
                  # Optional: Could also check response.success here if cost should only be counted for successes
                  all_costs.append(response.cost)
             # else: Skip entries without valid cost

        # Populate eval_results structure (currently only 'cost')
        eval_method = eval_methods[0] # Assuming only 'cost'
        eval_results.setdefault(eval_method, {})
        eval_results[eval_method]["scores"] = all_costs

        # Summarize collected costs
        if all_costs:
            summarized = ValuesSummarization(all_costs).summarize(summarization_methods)
            for idx, summarization_method in enumerate(summarization_methods):
                eval_results[eval_method][summarization_method] = summarized[idx]
        else:
            # Handle case with no valid costs
            for summarization_method in summarization_methods:
                eval_results[eval_method][summarization_method] = None # Use None for cost when no data

        # Ensure structure consistency for other potential future methods
        for method in eval_methods:
             if method not in eval_results:
                  eval_results[method] = {sm: None for sm in summarization_methods}
                  eval_results[method]["scores"] = []

        return eval_results

    @staticmethod
    def from_benchmark_model_results(
        benchmark_results,
        eval_methods=EVALUATION_METHODS,
        summarization_methods=ValuesSummarization.SUMMARIZATION_METHODS,
        summarize=True,
        handle_empty_results='error'
    ):
        """
        Calculates cost evaluation metrics across multiple benchmark model results.
        Uses the static from_benchmark_model_result method internally.
        """
        allowed_handling = ['error', 'ignore', 'zero']
        if handle_empty_results not in allowed_handling:
            raise ValueError(f"handle_empty_results must be one of {allowed_handling}, got '{handle_empty_results}'")

        # --- Use the updated from_benchmark_model_result --- #
        eval_results_per_model = []
        valid_benchmark_results = []
        for idx, benchmark_result in enumerate(benchmark_results):
             if hasattr(benchmark_result, 'model') and hasattr(benchmark_result, 'indexed_results'):
                 model_eval_results = CostMetrics.from_benchmark_model_result(
                     benchmark_result, eval_methods, summarization_methods
                 )
                 eval_results_per_model.append(model_eval_results)
                 valid_benchmark_results.append(benchmark_result)
             else:
                 print(f"Warning: Skipping benchmark result at index {idx} due to missing attributes (model or indexed_results).")

        if not summarize:
            return eval_results_per_model

        # --- Aggregation logic (similar to others, but handle_empty_results='zero' sets None) ---
        summarized_results = {}
        for eval_method in eval_methods:
            summarized_results[eval_method] = {}
            for summarization_method in summarization_methods:
                summarized_results[eval_method][summarization_method] = {}
                for idx, eval_result in enumerate(eval_results_per_model):
                    model_name = valid_benchmark_results[idx].model.name
                    metric_summary_value = eval_result.get(eval_method, {}).get(summarization_method)

                    # Check if value exists (could be None if no costs found)
                    # We store None if calculation resulted in None, otherwise the value
                    if eval_method in eval_result and summarization_method in eval_result.get(eval_method, {}):
                        summarized_results[eval_method][summarization_method][model_name] = metric_summary_value
                    else:
                        # Handle cases where the summary wasn't calculated (e.g., issue upstream)
                        if handle_empty_results == 'error':
                            raise KeyError(
                                f"Metric summary '{eval_method}' -> '{summarization_method}' not generated for model '{model_name}'. "
                                f"Check warnings from from_benchmark_model_result. Set handle_empty_results='ignore'/'zero'."
                            )
                        elif handle_empty_results == 'ignore':
                            pass
                        elif handle_empty_results == 'zero':
                            # Assign None as the summary statistic for cost when results are missing
                            summarized_results[eval_method][summarization_method][model_name] = None # Use None instead of 0.0 for cost

        return summarized_results