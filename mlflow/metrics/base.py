from dataclasses import dataclass
from mlflow.models.evaluation.base import (
    EvaluationMetric,
    make_metric,
)
from typing import Dict, List, Union
from mlflow.utils.class_utils import _get_class_from_string


@dataclass
class EvaluationExample:
    input_prompt: str
    output: str
    variables: Dict[str, str]
    score: float
    justification: str

    def toString(self) -> str:
        variables = ""
        for key, value in self.variables.items():
            variables += f"Provided {key}" + ":" + value + "\n"

        return f"""
      Question: {question}
      Provided answer:{provided_answer}
      {variables}
      Justification: {self.justification}
      Score: {self.score}
      """


@dataclass
class MetricValue:
    scores: List[float]
    justification: List[str]
    aggregate_result: Dict[str, float] = None


def make_genai_metric(
    name: str,
    version: str,
    definition: str,
    grading_prompt: str,
    examples: List[EvaluationExample] = None,
    model: str = "openai:/gpt4",
    variables: Dict[str, str] = None,
    parameters: Dict[str, any] = None,
    greater_is_better=True,
) -> EvaluationMetric:
    """
    This is a snippet of the code that is used to create a metric in MLflow.

    :param name: Name of the metric.
    :param version: Version of the metric.
    :param model: Model uri of the metric.
    :param variables: Variables required to compute the metric.
    :param definition: Definition of the metric.
    :param grading_prompt: Grading criteria of the metric.
    :param examples: Examples of the metric.
    :param parameters: Parameters for the llm used to compute the metric.
    :param greater_is_better: Whether the metric is better when it is greater.
    :return: A metric object.
    """
    import pandas
    import pyspark

    def eval_fn(
        eval_df: Union[pandas.Dataframe, pyspark.sql.DataFrame], builtin_metrics: Dict[str, float]
    ) -> MetricValue:
        """
        This is the function that is called when the metric is evaluated.
        """

        class_name = f"mlflow.metrics.utils.prompts.{version}.EvaluationModel"
        try:
            evaluation_model_class_module = _get_class_from_string(class_name)
        except Exception as e:
            if isinstance(e, ModuleNotFoundError):
                raise MlflowException(
                    f"Failed to find evaluation model for version {version}."
                    f"Please check the correctness of the version",
                    error_code=INVALID_PARAMETER_VALUE,
                ) from None
            else:
                raise MlflowException(
                    f"Failed to construct evaluation model {version}. Error: {e!r}",
                    error_code=INTERNAL_ERROR,
                ) from None

        for variable in variables:
            if variable in eval_df.columns:
                variable_value = eval_df[variable]
                variables_dict[variable] = variable_value
            else:
                print(f"{variable} does not exist in the DataFrame.")

        evaluation_context = evaluation_model_class_module(
            name,
            definition,
            grading_prompt,
            examples,
            model,
            variables_dict,
            parameters,
        )

        outputs = eval_df["prediction"]
        # Ask Corey how this is done
        inputs = eval_df["inputs"]
        model = evaluation_context.model
        parameters = evaluation_context.parameters
        grading_function = evaluation_context.grading_function

        # TODO: Save the metric definition in a yaml file

        messages = []
        for indx, _ in inputs.items():
            messages.append(
                [
                    {
                        "role": "system",
                        "content": evaluation_context.eval_prompt.partial_fill(
                            {
                                "input": inputs[indx],
                                "output": outputs[indx],
                            }
                        ),
                    }
                ],
            )

        # TODO: load the model and make the call
        return MetricValue(scores=[], justification=[])

    return make_metric(eval_fn, greater_is_better, name, version)
