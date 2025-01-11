from pydantic import BaseModel, Field


class TestCode(BaseModel):
    test_code: str = Field(
        description="Python test code to validate the functionality of the provided solution."
    )


class Code(BaseModel):
    """Schema for code solutions to questions."""

    prefix: str = Field(default="", description="Description of the model")
    imports: str = Field(
        default="", description="Code block import statements: Python 3.13 syntax"
    )
    code: str = Field(
        default="", description="Code block not including import statements"
    )


class Task(BaseModel):
    """Task to obtain module"""

    module_name: str = Field(default="", description="Module name")
    module_description: str = Field(default="", description="Module description")
    module_input_type: str = Field(
        default="", description="Type annotation of the module input"
    )
    module_output_type: str = Field(
        default="", description="Type annotation of the module output"
    )
    module_input_example: list = Field(default=[], description="Example input")
    module_output_example: list = Field(default=[], description="Example output")


class TaskList(BaseModel):  # organizer's formatter
    """List of Tasks"""

    tasks: list[Task] = Field(description="List of Task")
    overall_step: str = Field(description="Overall task steps")
    overall_task: Task = Field(description="Overall task information")


class GraphState(BaseModel):  # main graph's state
    """
    Represents the state of our graph.

    Attributes:
        messages: With user question, error messages, reasoning
        generation: Code solution
        task_details: list of Task
    """

    messages: list = Field(default=[], description="messages")
    user_problem: str = Field(default="", description="original user problem")
    generation: list[Code] = Field(default=[], description="code generated")
    task_details: list[Task] = Field(default=None, description="task details")
    overall_task: Task = Field(default=Task(), description="overall task")


class TaskState(BaseModel):  # code generator's subgraph state
    error: str = Field(default="", description="error or not")
    context: str = Field(default="", description="context info")
    iterations: int = Field(default=0, description="n-th number of iterations")
    code: Code = Field(default=Code(), description="list of codes")
    task: Task = Field(default=Task(), description="details of task")
