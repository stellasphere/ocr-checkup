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
        handle_empty_results='error'
    ):
        """
        Calculates string evaluation metrics across multiple benchmark model results.

        Args:
            benchmark_results: List of BenchmarkModelResult objects.
            eval_methods: List of evaluation methods to use (e.g., ['levenshtein_ratio']).
            summarization_methods: List of methods to summarize scores (e.g., ['average', 'median']).
            summarize (bool): If True, aggregates results by model name. If False, returns a list of results per model.
            handle_empty_results (str): How to handle models with no successful results for a metric.
                'error': Raise KeyError (default).
                'ignore': Skip the model for that metric's summary.
                'zero': Assign 0.0 for the summary statistics.

        Returns:
            dict or list: Summarized results (if summarize=True) or list of individual model results.
        """
        allowed_handling = ['error', 'ignore', 'zero']
        if handle_empty_results not in allowed_handling:
            raise ValueError(f"handle_empty_results must be one of {allowed_handling}, got '{handle_empty_results}'")

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

                    if eval_method in eval_result and summarization_method in eval_result.get(eval_method, {}):
                        summarized_results[eval_method][summarization_method][
                            model_name
                        ] = eval_result[eval_method][summarization_method]
                    else:
                        if handle_empty_results == 'error':
                            raise KeyError(
                                f"Metric summary '{eval_method}' -> '{summarization_method}' not found for model '{model_name}'. "
                                f"This often happens with 0 successful results. "
                                f"Set handle_empty_results='ignore' or handle_empty_results='zero' to change behavior."
                            )
                        elif handle_empty_results == 'ignore':
                            pass
                        elif handle_empty_results == 'zero':
                            summarized_results[eval_method][summarization_method][
                                model_name
                            ] = 0.0
                            
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
        handle_empty_results='error'
    ):
        """
        Calculates speed evaluation metrics across multiple benchmark model results.

        Args:
            benchmark_results: List of BenchmarkModelResult objects.
            eval_methods: List of evaluation methods to use (e.g., ['elapsed_time']).
            summarization_methods: List of methods to summarize scores (e.g., ['average', 'median']).
            summarize (bool): If True, aggregates results by model name. If False, returns a list of results per model.
            handle_empty_results (str): How to handle models with no successful results for a metric.
                'error': Raise KeyError (default).
                'ignore': Skip the model for that metric's summary.
                'zero': Assign 0.0 for the summary statistics.

        Returns:
            dict or list: Summarized results (if summarize=True) or list of individual model results.
        """
        allowed_handling = ['error', 'ignore', 'zero']
        if handle_empty_results not in allowed_handling:
            raise ValueError(f"handle_empty_results must be one of {allowed_handling}, got '{handle_empty_results}'")

        eval_results = []
        for benchmark_result in benchmark_results:
            eval_results.append(
                SpeedMetrics.from_benchmark_model_result(
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
                    if eval_method in eval_result and summarization_method in eval_result.get(eval_method, {}):
                        summarized_results[eval_method][summarization_method][
                            model_name
                        ] = eval_result[eval_method][summarization_method]
                    else:
                        if handle_empty_results == 'error':
                            raise KeyError(
                                f"Metric summary '{eval_method}' -> '{summarization_method}' not found for model '{model_name}'. "
                                f"This often happens with 0 successful results. "
                                f"Set handle_empty_results='ignore' or handle_empty_results='zero' to change behavior."
                            )
                        elif handle_empty_results == 'ignore':
                            pass
                        elif handle_empty_results == 'zero':
                            summarized_results[eval_method][summarization_method][
                                model_name
                            ] = 0.0

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
        handle_empty_results='error'
    ):
        """
        Calculates cost evaluation metrics across multiple benchmark model results.

        Args:
            benchmark_results: List of BenchmarkModelResult objects.
            eval_methods: List of evaluation methods to use (e.g., ['cost']).
            summarization_methods: List of methods to summarize scores (e.g., ['average', 'median']).
            summarize (bool): If True, aggregates results by model name. If False, returns a list of results per model.
            handle_empty_results (str): How to handle models with no cost data for a metric.
                'error': Raise KeyError (default).
                'ignore': Skip the model for that metric's summary.
                'zero': Assign None for the summary statistics (as cost=0 is ambiguous).

        Returns:
            dict or list: Summarized results (if summarize=True) or list of individual model results.
        """
        allowed_handling = ['error', 'ignore', 'zero']
        if handle_empty_results not in allowed_handling:
            raise ValueError(f"handle_empty_results must be one of {allowed_handling}, got '{handle_empty_results}'")

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
                    model_name = benchmark_results[idx].model.name
                    
                    if eval_method in eval_result and summarization_method in eval_result.get(eval_method, {}):
                        summarized_results[eval_method][summarization_method][
                            model_name
                        ] = eval_result[eval_method][summarization_method]
                    else:
                        # Case: Metric or summary was not calculated (likely due to zero results or no cost data)
                        if handle_empty_results == 'error':
                            raise KeyError(
                                f"Metric summary '{eval_method}' -> '{summarization_method}' not found for model '{model_name}'. "
                                f"This often happens with 0 successful results or missing cost data. "
                                f"Set handle_empty_results='ignore' or handle_empty_results='zero' to change behavior."
                            )
                        elif handle_empty_results == 'ignore':
                            # Do nothing, the model's entry for this metric/summary will be missing
                            pass
                        elif handle_empty_results == 'zero':
                            # Assign None as the summary statistic for cost when results are missing
                            summarized_results[eval_method][summarization_method][
                                model_name
                            ] = None # Use None instead of 0.0 for cost

        return summarized_results