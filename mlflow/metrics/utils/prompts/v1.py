from mlflow.metrics.base import (
    EvaluationExample,
)
from mlflow.metrics.utils.prompt_template import (
    PromptTemplate,
)

grading_system_prompt_template = PromptTemplate(
    """
    Please act as an impartial judge and evaluate the quality of the provided output which attempts to produce output for the provided input based on a provided information.
    You'll be given a function grading_function which you'll call for each provided information, input and provided output to submit your justification and score to compute the {name} of the output.

    Input:
    {input}

    Provided output:
    {output}

    {variables}

    Metric definition:
    {definition}

    Grading criteria:
    {grading_prompt}

    Examples:
    {examples}

    And you'll need to submit your grading for the {name} of the output, using the following format:
    Reasoning for {name}: [your step by step reasoning about the {name} of the output]
    Score for {name}: [your score number between 1 to 5 for the {name} of the output]
    """
)


class EvaluationModel:
    def __init__(
        self,
        name: str,
        definition: str,
        grading_prompt: str,
        examples: List[EvaluationExample] = None,
        model="openai:/gpt4",
        variables: List[str] = None,
        parameters: Dict[str, any] = None,
    ):
        self.name = name
        self.definition = definition
        self.grading_prompt = grading_prompt
        self.examples = examples
        self.model = model
        self.variables = variables
        self.parameters = parameters

    def to_dict(self):
        return {
            "model": self.model,
            "eval_prompt": grading_system_prompt_template.partial_fill(
                {
                    "name": self.name,
                    "definition": self.definition,
                    "grading_prompt": self.grading_prompt,
                    "examples": "\n".join([item.toString() for item in self.examples]),
                    "variables": "\n".join(
                        [
                            f"Provided {variable}: {variable_value}"
                            for variable, variable_value in self.variables
                        ]
                    ),
                }
            ),
            "parameters": self.parameters,
            "grading_function": self.get_openai_evaluator_function(),
        }

    def get_openai_evaluator_function(self):
        return {
            "name": "grading_function",
            "description": "Call this function to submit the grading for the output",
            "parameters": {
                "type": "object",
                "properties": {
                    f"reasoning_for_{self.name}": {
                        "type": "string",
                        "description": f"Your reasoning for giving the grading for the {self.name} of the output. Provide 5 to 30 words explanation.",
                    }
                },
                "required": [f"reasoning_for_{self.name}"],
            },
        }
