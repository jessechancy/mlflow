import re
from mlflow.metrics.base import (
    EvaluationExample,
)
from typing import Dict, List


class PromptTemplate:
    def __init__(self, template_str: str, variables: list = None):
        self.template_str = template_str
        if variables:
            self.variables = variables
        else:
            # Automatically parse variables from template string
            self.variables = re.findall(r"\{(\w+)\}", template_str)

    def format_prompt(self, **kwargs: dict):
        # Only keep the kwargs that are in the variables
        kwargs = {k: v for k, v in kwargs.items() if k in self.variables}

        # Format the prompt with the provided values
        return self.template_str.format(**kwargs)

    def partial_fill(self, **kwargs: dict):
        # Create a safe dictionary that returns the key if it doesn't exist in the dictionary
        safe_dict = {k: kwargs.get(k, "{" + k + "}") for k in self.variables}

        # Fill in the provided values, and return a new PromptTemplate
        new_template_str = self.template_str.format_map(safe_dict)
        unfilled_variables = [var for var in self.variables if var not in kwargs.keys()]
        return PromptTemplate(template_str=new_template_str, variables=unfilled_variables)

    # def fill_examples(self, examples: List[EvaluationExample]):
    #     examples_filled = "\n".join([item.toString() for item in examples])
    #     examples_key = "examples"

    #     # Create a safe dictionary that returns the key if it doesn't exist in the dictionary
    #     safe_dict = {k: "{" + k + "}" for k in self.variables}
    #     if "{" + examples_key + "}" in safe_dict:
    #         safe_dict["{" + examples_key + "}"] = examples_filled

    #     # Fill in the provided values, and return a new PromptTemplate
    #     new_template_str = self.template_str.format_map(safe_dict)
    #     if examples_key in self.variables:
    #         self.variables.remove(examples_key)
    #     return PromptTemplate(template_str=new_template_str, variables=self.variables)

    # def fill_variables(self, variables: Dict[str, str]):
    #     variables_filled = "\n".join(
    #         [
    #             f"Provided {variable}: variable_value"
    #             for variable, variable_value in variables.items()
    #         ]
    #     )
    #     variables_key = "variables"

    #     # Create a safe dictionary that returns the key if it doesn't exist in the dictionary
    #     safe_dict = {k: "{" + k + "}" for k in self.variables}
    #     if "{" + variables_key + "}" in safe_dict:
    #         safe_dict["{" + variables_key + "}"] = variables_filled

    #     # Fill in the provided values, and return a new PromptTemplate
    #     new_template_str = self.template_str.format_map(safe_dict)
    #     if variables_key in self.variables:
    #         self.variables.remove(variables_key)
    #     return PromptTemplate(template_str=new_template_str, variables=self.variables)
