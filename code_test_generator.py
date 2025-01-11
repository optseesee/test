from logging import getLogger
import io
import sys
from typing import Any, Callable
from unittest.mock import patch

import ipdb  # noqa
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables.base import RunnableSequence
from langgraph.graph import END, StateGraph
from langgraph.graph.state import CompiledStateGraph

import tools  # noqa
from states import Code, TaskState, TestCode
from tools import obtain_structured_llm_oai, strip_block_comment, code_to_str


def obtain_code_gen_prompt_oai() -> ChatPromptTemplate:
    """
    コード生成LLM用のプロンプトを生成します。

    Returns:
        ChatPromptTemplate: LLMで使用されるプロンプト。
    """
    system_message = strip_block_comment(
        """
        You are an expert Python programmer.
        Create Python code based on the following specifications.
        Structure your response as follows:

        1.  A brief description of the code.
        2.  The import statements (If program uses libraries. It can be empty.)
            **Do not** include function definitions in the import block.
            Type annotation syntax is based on Python 3.13.
            **Do NOT import `Dict`, `List`, `Tuple` from the `typing` module.**
            **Use built-in types `dict`, `list`, `tuple` directly in type hints.**
        3.  The functional code block.
            This code should use explicit type annotations as accurately as possible.
            Target Python version: 3.13, ensuring better code clarity and maintainability.
            Include comprehensive docstring focusing on detailed module descriptions.
            Do not include example usage (example: `if __name__ == "__main__":`).
            If input is from standard input, use the `input()` function.
            Input may contain line breaks.
        """
    )

    user_message_template = strip_block_comment(
        """
        # Module name
        {module_name}

        # Module description
        {module_description}

        # Input type (Python 3.13 syntax)
        {module_input_type}

        # Output type (Python 3.13 syntax)
        {module_output_type}

        # Input example
        {module_input_example}

        # Output example
        {module_output_example}

        {additional_info}
        """
    )
    return ChatPromptTemplate.from_messages(
        [
            ("system", system_message),
            ("user", user_message_template),
        ]
    )


def create_additional_info_string(
    context: str = "",
    code: str = "",
    error: str = "",
) -> str:
    """
    追加情報文字列を作成します。

    Args:
        context (str, optional): コンテキスト情報. Defaults to "".
        code (str, optional): 現在のコード. Defaults to "".
        error (str, optional): エラー情報. Defaults to "".

    Returns:
        str: 追加情報文字列。
    """

    info_parts = []
    if context:
        info_parts.append(f"# Additional information\n{context}")
    if code:
        info_parts.append(f"# Current code\n`python\n{code}\n`")
    if error:
        info_parts.append(f"# Error from the code above\n{error}")
    return "\n\n".join(info_parts)


def generate_code_generator(code_gen_chain: RunnableSequence) -> Callable:
    def code_generator(state: TaskState) -> dict[str, Any]:
        """
        Generate a code solution

        Args:
            state (TaskState): The current graph state

        Returns:
            state (TaskState): state updated with code, iterations
        """
        logger = getLogger(__name__).getChild("code_generator")

        task = state.task

        logger.info(f"---GENERATING CODE (iter: {state.iterations})---")

        code_solution = code_gen_chain.invoke(
            {
                "module_name": task.module_name,
                "module_description": task.module_description,
                "module_input_type": task.module_input_type,
                "module_output_type": task.module_output_type,
                "module_input_example": task.module_input_example,
                "module_output_example": task.module_output_example,
                "additional_info": create_additional_info_string(
                    context=state.context, code=state.code, error=state.error
                ),
            }
        )

        iterations = state.iterations + 1
        return {"code": code_solution, "iterations": iterations}

    return code_generator


def code_check(state: TaskState):
    """
    Check code

    Args:
        state (TaskState): The current graph state

    Returns:
        state (TaskState): New key added to state, error
    """
    logger = getLogger(__name__).getChild("code_check")

    logger.info("---CHECKING CODE FORMAT---")

    # State
    code_solution = state.code

    # Get solution components
    task = state.task
    imports = code_solution.imports

    # Check imports
    try:
        exec(imports)
        logger.info("---Import check PASS---")
    except Exception as e:
        logger.warning("---CODE IMPORT CHECK: FAILED---")
        logger.warning(e)
        ipdb.set_trace()
        return {"error": e}

    try:
        full_code = code_to_str(code_solution)  # stuck_codes(imports, code)
        for ex_input in task.module_input_example:
            with patch("builtins.input", return_value=ex_input):
                exec(full_code, globals())
        logger.info("---Execute PASS---")
    except Exception as e:
        logger.warning("---CODE BLOCK CHECK: FAILED---")
        logger.warning(e)
        ipdb.set_trace()
        return {"error": e}

    # No errors
    logger.info("---NO CODE FORMAT TEST FAILURES---")
    return {"error": ""}


def obtain_test_case_gen_prompt():
    comment = strip_block_comment(
        """
        Your task is to create unit tests for the provided Python code using the given test cases.

        ### Target Code:
        {code}

        ### Provided Test Cases:
        You are provided with multiple test cases, each consisting of an input and an expected output.
        Use these test cases to generate Python test code:
        - **Inputs:**
        {module_input_example}
        - **Expected Outputs:**
        {module_output_example}

        Each input corresponds to its respective expected output by position in the list.

        ### Requirements:
        1. Use a Python testing framework such as `unittest` to generate the test code.
        2. Write test cases that strictly follow the provided inputs and outputs,
           without introducing new assumptions or cases.
        3. Ensure that all inputs and expected outputs are covered in the test code.
        4. Mock any external dependencies if required to ensure isolated and repeatable tests.
        5. Ensure that the generated test code is ready for execution and adheres to standard practices.

        ### Execution-Specific Requirement:
        6. Ensure that the generated test code can be executed dynamically using `exec()` in Python.
        - Include `unittest.main(module=__name__, exit=False)` in the `if __name__ == "__main__":` block.
          This ensures that the program does not terminate after running the tests.
        - Ensure that all test cases are defined in the same code block to avoid external dependencies.
        - Make sure that the test code runs correctly in the context of dynamic evaluation via `exec()` and
          does not cause the program to terminate.

        ### Output:
        Provide the test code only, with the following structure:
        1. Import necessary modules and frameworks.
        2. Define the test class and methods, with one method per test case.
        3. Ensure that each test method corresponds to one input-output pair.
        4. Do not include any additional explanations or comments.

        """
    )
    return ChatPromptTemplate.from_messages(
        [
            ("system", comment),
        ]
    )


def test_generator(state: TaskState):
    """
    Generate test cases using AI and test the generated code.

    Args:
        state (TaskState): The current graph state
        test_case_gen_fn: Function for generating test cases
        concatenated_content: Context for generating test cases

    Returns:
        state (TaskState): Updated state with test results
    """
    logger = getLogger(__name__).getChild("test_generator")
    logger.info("---TESTING CODE WITH AI-GENERATED TEST CASES---")

    # State
    task = state.task
    module_input_example = task.module_input_example
    module_output_example = task.module_output_example
    # imports = state.code.imports
    # code = state.code.code

    generated_code = code_to_str(state.code)

    # Generate test cases using AI
    try:
        test_structured_llm_oai = obtain_structured_llm_oai(
            output_structure=TestCode, prompt=obtain_test_case_gen_prompt()
        )

        logger.info("---Generated Test Cases---")
        response = test_structured_llm_oai.invoke(
            {
                "code": generated_code,
                "module_input_example": module_input_example,
                "module_output_example": module_output_example,
            }
        )
        test_code = response["parsed"].test_code

    except Exception as e:
        logger.warning("---TEST CASE GENERATION FAILED---")
        logger.warning(e)
        return {"error": e}

    # Execute generated test code
    # テストコードと元のコードを連結
    exec_test_code = f"{generated_code}\n{test_code}"

    # 標準出力をリダイレクト
    stream = io.StringIO()
    sys.stderr = stream

    exec(exec_test_code, globals())

    # 標準出力の内容を取得
    sys.stderr = sys.__stderr__  # リダイレクト解除
    output = stream.getvalue()

    # 実行エラー・失敗/成功を判定
    error_keywords = {"ERROR", "Error", "FAILED", "FAIL"}
    # if "ERROR" in output or "Error" or "FAILED" or "FAIL" in output:
    if any(keyword in output for keyword in error_keywords):
        logger.warning("---TEST FAILED---")
        logger.warning(f"{output=}")
        return {"error": output}

    logger.info("---Test PASS---")
    return {"error": ""}


def generate_reflect(code_gen_chain: RunnableSequence):
    def reflect(state: TaskState):
        """
        reflect on errors

        Args:
            state (TaskState): The current graph state

        Returns:
            state (TaskState): New key added to state, generation
        """
        logger = getLogger(__name__).getChild("reflect")

        # State
        iterations = state.iterations
        task_details = state.task

        logger.info(f"---GENERATING CODE SOLUTION (iter: {iterations})---")

        # Prompt reflection

        # Add reflection
        reflections = code_gen_chain.invoke(
            {
                "module_name": task_details.module_name,
                "module_description": task_details.module_description,
                "module_input_type": task_details.module_input_type,
                "module_output_type": task_details.module_output_type,
                "module_input_example": task_details.module_input_example,
                "module_output_example": task_details.module_output_example,
                "additional_info": create_additional_info_string(
                    state.error,
                    state.code,
                    state.context,
                ),
            }
        )
        messages = [("assistant", f"Here are reflections on the error: {reflections}")]
        return {"messages": messages, "iterations": iterations}

    return reflect


def insert_errors(inputs):
    """Insert errors for tool parsing in the messages"""

    # Get errors
    error = inputs["error"]
    messages = inputs["messages"]
    messages += [
        (
            "assistant",
            f"Retry. You are required to fix the parsing errors: {error} \n\n You must invoke the provided tool.",
        )
    ]
    return {
        "messages": messages,
        "context": inputs["context"],
    }


def parse_output(solution):
    return solution["parsed"]


def generate_decide_to_finish(max_iterations: int):
    def decide_to_finish(state: TaskState):
        """
        Determins whether to finish.

        Args:
            state (TaskState): The current graph state

        Returns:
            str: Next node to call
        """
        logger = getLogger(__name__).getChild("decide_to_finish")

        error = state.error
        iterations = state.iterations

        if error == "":
            logger.info("---DECISION: COMPLETE---")
            return "end"
        if iterations == max_iterations:
            logger.warning("---DECISION: GAVE-UP---")
            return "end"
        logger.info("---DECISION: RE-TRY SOLUTIONS---")
        return "reflect"

    return decide_to_finish


def generate_code_generation_graph(
    chain: RunnableSequence = obtain_structured_llm_oai(
        output_structure=Code, prompt=obtain_code_gen_prompt_oai()
    )
) -> CompiledStateGraph:
    fallback_chain = insert_errors | chain

    N = 3  # Max re-tries
    code_gen_chain_re_try = chain.with_fallbacks(
        fallbacks=[fallback_chain] * N, exception_key="error"
    )

    code_gen_chain_re_try_out = code_gen_chain_re_try | parse_output

    code_generation_subgraph = StateGraph(TaskState)

    code_generation_subgraph.add_node(
        "code_generator", generate_code_generator(code_gen_chain_re_try_out)
    )
    code_generation_subgraph.add_node("code_check", code_check)
    code_generation_subgraph.add_node(
        "reflect", generate_reflect(code_gen_chain_re_try_out)
    )
    code_generation_subgraph.add_node("test_generator", test_generator)
    code_generation_subgraph.add_edge("code_generator", "code_check")
    code_generation_subgraph.add_conditional_edges(
        "code_check",
        generate_decide_to_finish(N),
        {"end": "test_generator", "reflect": "reflect"},
    )
    code_generation_subgraph.add_conditional_edges(
        "test_generator",
        generate_decide_to_finish(N),
        {"end": END, "reflect": "reflect"},
    )
    code_generation_subgraph.add_edge("reflect", "code_generator")

    code_generation_subgraph.set_entry_point("code_generator")

    code_generation_subgraph.graph_attrs = {"compound": "true"}

    code_generation_app = code_generation_subgraph.compile()

    code_generation_app.name = "code_generation_subgraph"

    return code_generation_app
