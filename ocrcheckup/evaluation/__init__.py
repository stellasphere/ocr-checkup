import Levenshtein
import statistics
from typing import List, Any


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

    def from_benchmark_model_result(
        benchmark_result,
        eval_methods=EVALUATION_METHODS,
        summarization_methods=ValuesSummarization.SUMMARIZATION_METHODS,
    ):
        eval_results = {}
        for idx, result in enumerate(benchmark_result.results):
            ground_truth = benchmark_result.benchmark.annotations[idx]

            result_evaluated = StringMetrics(result, ground_truth).evaluate(
                eval_methods
            )
            for method_idx, method_score in enumerate(result_evaluated):
                eval_method = eval_methods[method_idx]
                eval_results.setdefault(eval_method, {})
                eval_results[eval_method].setdefault("scores", [])
                eval_results[eval_method]["scores"].append(method_score)

        for eval_method, eval_result in eval_results.items():
            summarized = ValuesSummarization(eval_result["scores"]).summarize(
                summarization_methods
            )

            for idx, summarization_method in enumerate(summarization_methods):
                eval_results[eval_method][summarization_method] = summarized[idx]

        return eval_results

    def from_benchmark_model_results(
        benchmark_results,
        eval_methods=EVALUATION_METHODS,
        summarization_methods=ValuesSummarization.SUMMARIZATION_METHODS,
        summarize=True,
    ):
        eval_results = []
        for idx, benchmark_result in enumerate(benchmark_results):
            eval_results.append(
                StringMetrics.from_benchmark_model_result(
                    benchmark_result, eval_methods, summarization_methods
                )
            )

        if summarize is False:
            return eval_results

        summarized_results = {}
        for eval_method in eval_methods:
            summarized_results[eval_method] = {}
            for summarization_method in summarization_methods:
                summarized_results[eval_method][summarization_method] = {}
                for idx, eval_result in enumerate(eval_results):
                    model_name = benchmark_results[idx].model.name
                    summarized_results[eval_method][summarization_method][
                        model_name
                    ] = eval_result[eval_method][summarization_method]

        return summarized_results
    

class SpeedMetrics:
    EVALUATION_METHODS = ["elapsed_time"]

    def __init__(self, elapsed_times):
        self.elapsed_times = elapsed_times

    def elapsed_time(self):
        return self.elapsed_times

    def evaluate(self, methods=EVALUATION_METHODS):
        return tuple(getattr(self, method)() for method in methods)

    def from_benchmark_model_result(
        benchmark_result,
        eval_methods=EVALUATION_METHODS,
        summarization_methods=ValuesSummarization.SUMMARIZATION_METHODS,
    ):
        eval_results = {}
        for idx, result in enumerate(benchmark_result.elapsed_times):

            result_evaluated = SpeedMetrics(result).evaluate(
                eval_methods
            )
            for method_idx, method_score in enumerate(result_evaluated):
                eval_method = eval_methods[method_idx]
                eval_results.setdefault(eval_method, {})
                eval_results[eval_method].setdefault("scores", [])
                eval_results[eval_method]["scores"].append(method_score)

        for eval_method, eval_result in eval_results.items():
            summarized = ValuesSummarization(eval_result["scores"]).summarize(
                summarization_methods
            )

            for idx, summarization_method in enumerate(summarization_methods):
                eval_results[eval_method][summarization_method] = summarized[idx]

        return eval_results

    def from_benchmark_model_results(
        benchmark_results,
        eval_methods=EVALUATION_METHODS,
        summarization_methods=ValuesSummarization.SUMMARIZATION_METHODS,
        summarize=True,
    ):
        eval_results = []
        for benchmark_result in benchmark_results:
            eval_results.append(
                SpeedMetrics.from_benchmark_model_result(
                    benchmark_result, eval_methods, summarization_methods
                )
            )

        if not summarize:
            return eval_results

        summarized_results = {}
        for eval_method in eval_methods:
            summarized_results[eval_method] = {}
            for summarization_method in summarization_methods:
                summarized_results[eval_method][summarization_method] = {}
                for idx, eval_result in enumerate(eval_results):
                    model_name = benchmark_results[idx].model.name
                    summarized_results[eval_method][summarization_method][
                        model_name
                    ] = eval_result[eval_method][summarization_method]

        return summarized_results


class CostMetrics:
    EVALUATION_METHODS = ["cost"]

    @staticmethod
    def cost(cost_value):
        # Simple method to potentially transform cost if needed in the future
        # For now, just returns the value if not None
        return cost_value

    @staticmethod
    def from_benchmark_model_result(
        benchmark_result,
        eval_methods=EVALUATION_METHODS,
        summarization_methods=ValuesSummarization.SUMMARIZATION_METHODS,
    ):
        eval_results = {}
        # Access raw results which contain OCRModelResponse objects
        raw_results = benchmark_result.results_raw 
        costs = [res.cost for res in raw_results if res.success and res.cost is not None]
        
        # Handle case where no successful results with cost exist
        if not costs:
             for eval_method in eval_methods:
                eval_results[eval_method] = {sm: None for sm in summarization_methods}
                eval_results[eval_method]["scores"] = []
             return eval_results


        # Currently only one method: "cost"
        eval_method = eval_methods[0]
        eval_results.setdefault(eval_method, {})
        eval_results[eval_method]["scores"] = costs

        # Summarize the collected costs
        summarized = ValuesSummarization(costs).summarize(summarization_methods)
        for idx, summarization_method in enumerate(summarization_methods):
            eval_results[eval_method][summarization_method] = summarized[idx]
        
        # Ensure structure consistency even if other methods were requested (though only 'cost' is supported)
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
    ):
        eval_results = []
        for benchmark_result in benchmark_results:
            eval_results.append(
                CostMetrics.from_benchmark_model_result(
                    benchmark_result, eval_methods, summarization_methods
                )
            )

        if not summarize:
            return eval_results

        # Aggregate results by model
        summarized_results = {}
        for eval_method in eval_methods:  # Should typically just be ['cost']
            summarized_results[eval_method] = {}
            for summarization_method in summarization_methods:
                summarized_results[eval_method][summarization_method] = {}
                for idx, eval_result in enumerate(eval_results):
                    # Handle cases where a model might not have cost results (e.g., all failed)
                    if eval_method in eval_result and summarization_method in eval_result[eval_method]:
                        model_name = benchmark_results[idx].model.name
                        summarized_results[eval_method][summarization_method][
                            model_name
                        ] = eval_result[eval_method][summarization_method]
                    else: 
                        # Ensure model entry exists even if no valid cost data
                        model_name = benchmark_results[idx].model.name
                        summarized_results[eval_method][summarization_method][model_name] = None 

        return summarized_results


def calculate_efficiency_metrics(
    string_metrics_results: dict,
    speed_metrics_results: dict,
    cost_metrics_results: dict,
    accuracy_method: str = "levenshtein_ratio",
    accuracy_summary: str = "median",
    speed_summary: str = "median",
    cost_summary: str = "median",
) -> dict:
    """Calculates speed and cost efficiency metrics.

    Args:
        string_metrics_results: Results from StringMetrics.from_benchmark_model_results.
        speed_metrics_results: Results from SpeedMetrics.from_benchmark_model_results.
        cost_metrics_results: Results from CostMetrics.from_benchmark_model_results.
        accuracy_method: The string metric to use for accuracy (e.g., 'levenshtein_ratio').
        accuracy_summary: The summary statistic for accuracy (e.g., 'median', 'average').
        speed_summary: The summary statistic for speed (e.g., 'median', 'average').
        cost_summary: The summary statistic for cost (e.g., 'median', 'average').

    Returns:
        A dictionary containing 'speed_efficiency' and 'cost_efficiency',
        each mapping model names to their calculated efficiency.
    """
    efficiency_results = {"speed_efficiency": {}, "cost_efficiency": {}}

    # Ensure the specified metrics/summaries exist
    if accuracy_method not in string_metrics_results:
        raise ValueError(f"Accuracy method '{accuracy_method}' not found in string metrics results.")
    if accuracy_summary not in string_metrics_results[accuracy_method]:
        raise ValueError(f"Accuracy summary '{accuracy_summary}' not found for method '{accuracy_method}'.")
    
    if "elapsed_time" not in speed_metrics_results:
         raise ValueError("'elapsed_time' not found in speed metrics results.")
    if speed_summary not in speed_metrics_results["elapsed_time"]:
        raise ValueError(f"Speed summary '{speed_summary}' not found for 'elapsed_time'.")

    if "cost" not in cost_metrics_results:
         raise ValueError("'cost' not found in cost metrics results.")
    if cost_summary not in cost_metrics_results["cost"]:
         raise ValueError(f"Cost summary '{cost_summary}' not found for 'cost'.")


    accuracy_data = string_metrics_results[accuracy_method][accuracy_summary]
    speed_data = speed_metrics_results["elapsed_time"][speed_summary]
    cost_data = cost_metrics_results["cost"][cost_summary]

    all_models = set(accuracy_data.keys()) | set(speed_data.keys()) | set(cost_data.keys())

    for model_name in all_models:
        accuracy = accuracy_data.get(model_name)
        speed = speed_data.get(model_name)
        cost = cost_data.get(model_name)

        # Calculate Speed Efficiency
        if accuracy is not None and speed is not None and speed > 0:
            efficiency_results["speed_efficiency"][model_name] = accuracy / speed
        else:
            efficiency_results["speed_efficiency"][model_name] = None  # Indicate calculation not possible

        # Calculate Cost Efficiency
        if accuracy is not None and cost is not None and cost > 0:
            efficiency_results["cost_efficiency"][model_name] = accuracy / cost
        else:
            efficiency_results["cost_efficiency"][model_name] = None # Indicate calculation not possible

    return efficiency_results
