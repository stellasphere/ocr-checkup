import json
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, Optional, Union, List


class CostType(Enum):
    EXTERNAL = auto() # Cost based on external API calls (e.g., OpenAI, Anthropic)
    COMPUTE = auto()  # Cost based on local compute resources (e.g., GPU time)
    UNKNOWN = auto()  # Cost cannot be determined



@dataclass
class ModelCost:
    """
    Represents the *inputs* required to calculate the cost of a model inference.

    Attributes:
        cost_type: The basis for the cost calculation (EXTERNAL, COMPUTE, UNKNOWN).
        info: A dictionary containing necessary information for cost calculation.
              For EXTERNAL: Expected keys: 'model_id': str, 'input_tokens': int, 'output_tokens': int.
              For COMPUTE: Expected keys: 'runtime_seconds': float.
              Actual keys depend on the specific model and how it's run.
    """
    cost_type: CostType
    info: Dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        return f"ModelCost(type={self.cost_type.name}, info={self.info})"

    def __repr__(self) -> str:
        return f"ModelCost(cost_type={self.cost_type}, info={self.info!r})"


# --- New ModelPricing Class --- #
@dataclass
class ModelPricing:
    """
    Holds the specific pricing rates for a given model or service.

    Attributes are dynamically assigned based on keyword arguments provided
    during instantiation. These attribute names should correspond to keys
    expected in the `ModelCost.info` dictionary for quantity information.

    Examples:
        ModelPricing(input_token=0.000005, output_token=0.000015)
        ModelPricing(page=0.001)
        ModelPricing(api_call=0.50, character=0.0001)
    """
    def __init__(self, **kwargs):
        # Dynamically assign attributes based on kwargs
        for key, value in kwargs.items():
            try:
                # Ensure rates are stored as floats
                setattr(self, key, float(value))
            except (ValueError, TypeError):
                raise ValueError(f"Pricing rate for '{key}' must be a number, got: {value}")
        if not kwargs:
             raise ValueError("ModelPricing must be initialized with at least one pricing rate.")

    def __str__(self) -> str:
        rates = ", ".join(f"{k}={v}" for k, v in vars(self).items())
        return f"ModelPricing({rates})"

    def __repr__(self) -> str:
        return f"ModelPricing(**{vars(self)!r})"


class ModelCostCalculator:
    """
    Calculates the monetary cost of model inferences based on ModelCost data.

    Uses provided pricing data (ModelPricing instances) for external/variable rate models
    and compute cost parameters for local execution.

    Attributes:
        pricing_data (Optional[Dict[str, ModelPricing]]): Dictionary mapping model IDs
                                        to their ModelPricing instances.
                                        Required for calculating EXTERNAL costs.
        compute_cost_params (Optional[Dict]): Dictionary with parameters for compute
                                              cost (e.g., 'cost_per_second').
                                              Required for calculating COMPUTE costs.
    """
    def __init__(
        self,
        # Updated type hint for pricing_data
        pricing_data: Optional[Dict[str, ModelPricing]] = None,
        compute_cost_params: Optional[Dict[str, Union[float, str]]] = None
    ):
        # Ensure pricing_data contains ModelPricing instances
        if pricing_data:
            for model_id, pricing in pricing_data.items():
                if not isinstance(pricing, ModelPricing):
                    raise TypeError(f"pricing_data must contain ModelPricing instances, but found {type(pricing)} for key '{model_id}'")
        self.pricing_data = pricing_data or {}
        self.compute_cost_params = compute_cost_params or {}
        print(f"ModelCostCalculator initialized. # Pricing models: {len(self.pricing_data)}")

    def calculate_single_cost(self, model_cost: ModelCost) -> Optional[float]:
        """
        Calculates the cost for a single ModelCost instance.

        Args:
            model_cost: The ModelCost object containing runtime information.

        Returns:
            The calculated cost as a float, or None if calculation fails.
            Assumes USD currency based on typical API pricing conventions.
        """
        try:
            if model_cost.cost_type == CostType.EXTERNAL:
                # Renamed internal method call
                return self._calculate_variable_rate_cost(model_cost.info)
            elif model_cost.cost_type == CostType.COMPUTE:
                return self._calculate_compute_cost(model_cost.info)
            elif model_cost.cost_type == CostType.UNKNOWN:
                raise ValueError(f"Cannot calculate cost for ModelCost with UNKNOWN type. Info: {model_cost.info}")
            else:
                # Should not happen with Enum
                raise ValueError(f"Unsupported cost_type encountered: {model_cost.cost_type}")
        except ValueError as e:
             # Catch calculation errors (missing data, invalid types) from internal methods
             raise ValueError(f"Cost calculation failed for {model_cost}: {e}")
        except Exception as e:
            # Catch unexpected errors
            raise ValueError(f"Unexpected error during cost calculation for {model_cost}: {e}")

    # Renamed from _calculate_external_cost
    def _calculate_variable_rate_cost(self, info: Dict[str, Any]) -> float:
        """Calculates cost for models priced by variable rates defined in ModelPricing."""
        model_id = info.get("model_id")

        if not model_id:
            raise ValueError(f"Missing 'model_id' in info for EXTERNAL cost calculation. Info: {info}")

        if not self.pricing_data:
            raise ValueError(f"Attempted EXTERNAL cost calculation for '{model_id}', but no pricing_data provided to the calculator.")

        model_pricing: Optional[ModelPricing] = self.pricing_data.get(model_id)
        if not model_pricing:
            raise ValueError(f"Pricing data (ModelPricing instance) not found for model_id: '{model_id}' in the calculator's pricing_data.")

        total_cost = 0.0
        # Iterate through the rates defined in the ModelPricing object
        pricing_rates = vars(model_pricing)
        if not pricing_rates:
             # Should not happen due to ModelPricing validation, but good practice
             raise ValueError(f"ModelPricing for '{model_id}' has no rates defined.")

        for rate_name, rate_value in pricing_rates.items():
            # Get the corresponding quantity from the ModelCost info
            quantity = info.get(rate_name)

            if quantity is None:
                raise ValueError(f"Missing quantity key '{rate_name}' in ModelCost.info for model '{model_id}'. Required by its ModelPricing. Info: {info}")

            # Validate quantity type and value
            try:
                 quantity_float = float(quantity)
                 if quantity_float < 0:
                     raise ValueError(f"Quantity for '{rate_name}' cannot be negative.")
            except (ValueError, TypeError) as e:
                 raise ValueError(f"Invalid quantity for '{rate_name}' for model '{model_id}': value='{quantity}'. Error: {e}")

            # Calculate cost component and add to total
            try:
                cost_component = quantity_float * rate_value # rate_value is already float
                total_cost += cost_component
            except Exception as e:
                # Should be unlikely given prior checks, but catch potential issues
                raise ValueError(f"Error calculating cost component for '{rate_name}' in model '{model_id}': {e}")

        return total_cost

    def _calculate_compute_cost(self, info: Dict[str, Any]) -> float:
        """Calculates cost based on compute resources."""
        runtime = info.get("runtime_seconds")

        if runtime is None:
            raise ValueError("Missing 'runtime_seconds' in info for COMPUTE cost calculation.")

        if not self.compute_cost_params:
            raise ValueError("Missing 'compute_cost_params' in calculator for COMPUTE cost calculation.")

        cost_per_second = self.compute_cost_params.get("cost_per_second")
        if cost_per_second is None:
            raise ValueError("Missing 'cost_per_second' in calculator's compute_cost_params for COMPUTE cost calculation.")

        try:
            runtime_float = float(runtime)
            cost_per_second_float = float(cost_per_second)
            if runtime_float < 0:
                 raise ValueError("runtime_seconds cannot be negative")
            if cost_per_second_float < 0:
                 raise ValueError("cost_per_second cannot be negative")

            cost = runtime_float * cost_per_second_float
            return cost
        except (ValueError, TypeError) as e:
            raise ValueError(f"Invalid compute cost input: runtime='{runtime}', cost_per_second='{cost_per_second}'. Error: {e}")

    def calculate_batch_cost(self, model_costs: List[ModelCost]) -> List[Optional[float]]:
        """
        Calculates the cost for a batch of ModelCost instances.

        Args:
            model_costs: A list of ModelCost instances.

        Returns:
            A list of calculated costs, or None for each ModelCost instance in the input list.
        """
        return [self.calculate_single_cost(mc) for mc in model_costs]

    @staticmethod
    def import_litellm_costs():
        # Get JSON from https://raw.githubusercontent.com/BerriAI/litellm/refs/heads/main/model_prices_and_context_window.json
        import requests
        import json

        url = "https://raw.githubusercontent.com/BerriAI/litellm/refs/heads/main/model_prices_and_context_window.json"
        response = requests.get(url)
        response.raise_for_status()

        # Parse the JSON
        litellm_data = json.loads(response.text)

        # Create a dictionary mapping model names to their pricing data
        pricing_data = {}
        for model_name, model_info in litellm_data.items():
            # Remap model pricing to match the ModelPricing class
            model_pricing = {}
            for key, value in model_info.items():
                if key == "input_cost_per_token":
                    model_pricing["input_tokens"] = value
                elif key == "output_cost_per_token":
                    model_pricing["output_tokens"] = value
            
            if len(model_pricing.keys()) > 0:
                pricing_data[model_name] = ModelPricing(**model_pricing)

        return pricing_data
        
    @staticmethod
    def import_default_costs():
        default_pricing = {
            "mistral-ocr-2503": {
                "pages": 1/1000
            },
            "roboflow-serverless": {
                "elapsed_time": (1/500)*3 # 1 credit for 500 seconds, 3 dollars per credit
            }
        }

        model_pricing_dict = {}
        for model_id, pricing_info in default_pricing.items():
            model_pricing_dict[model_id] = ModelPricing(**pricing_info)

        return model_pricing_dict


    @staticmethod
    def default():
        # Combine LiteLLM and default pricing info
        litellm_data = ModelCostCalculator.import_litellm_costs()
        default_pricing = ModelCostCalculator.import_default_costs()

        pricing_data = {**litellm_data, **default_pricing}

        return ModelCostCalculator(pricing_data=pricing_data)

        

