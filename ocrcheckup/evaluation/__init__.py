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
