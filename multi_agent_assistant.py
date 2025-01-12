import argparse
import tempfile
from logging import getLogger
from pathlib import Path
from typing import Callable
import os
import ipdb  # noqa
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables.base import RunnableSequence
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

import tools  # noqa
from code_test_generator import generate_code_generation_graph
from states import Code, GraphState, TaskList, TaskState
from tools import (
    codes_to_str,
    merge_and_format_code,
    obtain_structured_llm_oai,
    plot_graph,
    strip_block_comment,
)
from utils import set_base_log_level, set_log_level

#langsmith見方
#https://zenn.dev/pharmax/articles/61edc477e4de17
os.environ["LANGCHAIN_TRACING_V2"]="true"
os.environ["LANGCHAIN_ENDPOINT"]="https://api.smith.langchain.com"
os.environ["LANGCHAIN_API_KEY"]="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"
os.environ["LANGCHAIN_PROJECT"]="multi_agent_dev1"

def _retrieve_args():
    """
    このツールの実行時標準入力のハンドリングのための関数。
    python multi_agent_assistant.py -h
    でヘルプ表示。
    """
    parser = argparse.ArgumentParser(
        description="topse Multi-Agent App",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--ask",
        type=str,
        help="directry ask question (highest priority)",
    )
    parser.add_argument(
        "--question_file",
        type=Path,
        help="input question markdown file path (2nd highest priority)",
    )
    parser.add_argument(
        "--question_dir",
        type=Path,
        default=Path("question_files"),
        help="input question file directory",
    )
    parser.add_argument(
        "--question_id",
        type=str,
        default="ABC201C",
        help="input question ID (usual use)",
    )
    parser.add_argument(
        "--interact",
        action="store_true",
        help="enter interactive debug mode at last",
    )
    parser.add_argument(
        "--no_graph_plot",
        dest="graph_plot",
        action="store_false",
        help="not plot for LangGraph by this option"
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=1,
        help="increase output verbosity",
    )
    parser.add_argument(
        "-q",
        "--quiet",
        action="count",
        default=0,
        help="decrease output verbosity",
    )
    args = parser.parse_args()
    args.verbose -= args.quiet
    del args.quiet

    if args.ask:
        with tempfile.NamedTemporaryFile(mode="w", delete=False) as t:
            t.write(args.ask)
            args.question_file = Path(t.name)
            print(f"your question:\n{args.ask}")

    del args.ask

    if not args.question_file:
        args.question_file = args.question_dir / f"{args.question_id}.md"

    del args.question_dir
    del args.question_id

    if not args.question_file.is_file():
        raise FileNotFoundError(f"{args.question_file} NOT FOUND")

    return args


def obtain_organizer_prompt_oai():
    system_comment = strip_block_comment(
        """
        You are an experienced team leader.
        When you receive a programming problem from a user,
        create a clear step-by-step plan (algorithm design) to solve the problem.
        Consider each step as an independent, reusable module.

        Provide followings:
        *   detailed overall steps to solve the problem (listed description is expected)
        *   overall task information
        *   list of task informations of each step

        Strictly provide the following information for `overall task` and `each task`:

        *   **Module Name:** A concise and easy-to-understand
            module name (snake_case is recommended).
        *   **Task Description:** A detailed description of what
            the module does (concisely but clearly).
        *   **Input:** The input to the module and its type.
            Use Python type annotation syntax, considering Python 3.13 features.
        *   **Output:** The output from the module and its type.
            Use Python type annotation syntax, considering Python 3.13 features.
        *   **Input Example:** Example of set of the input
        *   **Output Example:** Example of set of the output drived from the Input Example

        Provide the plan in a numbered list format.
        If multiple solutions are possible, provide each plan in a separate list.
        Describe the dependencies between modules clearly.

        **Example:**

        **Problem:** Calculate the average of a list of numbers (integers or floats).
        Handle cases where the input list is empty.

        **Plan:**

        1.  **Module Name:** `validate_input`
            *   **Task Description:** Checks if the input is a
                list and if all elements are either integers
                or floats. Handles empty list input.
            *   **Input:** `input: Any`
            *   **Output:** `output: bool`
                (Returns False if the input is invalid or empty)
            *   **Input Example:** `[[], [6, 10, 20], "abc"]`
            *   **Output Example:** `[False, True, False]`

        2.  **Module Name:** `calculate_sum`
            *   **Task Description:** Calculates the sum of a list of numbers.
            *   **Input:** `numbers: list[int | float]`
            *   **Output:** `output: int | float`
            *   **Input Example:** `[[3], [6, 10, 20], [2.0, 1, 2.1]]`
            *   **Output Example:** `[3, 36, 5.1]`

        3.  **Module Name:** `calculate_average`
            *   **Task Description:** Calculates the average from the sum of numbers.
            *   **Input:** `{{"total": int | float, "count": int}}`
            *   **Output:** `average: float | None` (Returns None if count is 0 to avoid division by zero)
            *   **Input Example:**
                `[{{"total": 3, "count": 1}}, {{"total": 35, "count": 3}}, {{"total": 5.1, "count": 3}}]`
            *   **Output Example:** `[3, 12, 1.7]`

        **overall task:**
        -   **Module Name:** `calculate_average_of_numbers`
            *   **Task Description:** Calculates the average of a list of numbers.
            *   **Input:** `numbers: list[int | float]`
            *   **Output:** `output: int | float`
            *   **Input Example:** `[[3], [6, 10, 20], [2.0, 1, 2.1]]`
            *   **Output Example:** `[3, 12, 1.7]`
        """
    )

    return ChatPromptTemplate.from_messages(
        [
            ("system", system_comment),
            ("placeholder", "{messages}"),
        ]
    )


def generate_organizer(organizer_chain: RunnableSequence):
    def organizer(state: GraphState):
        """
        Generate a code solution

        Args:
            state (GraphState): The current graph state

        Returns:
            state (GraphState): New key added to state
        """
        logger = getLogger(__name__).getChild("organizer")

        logger.info("---HANDLING TASK @ ORGANIZER---")

        messages = state.messages

        # Solution
        response_org = organizer_chain.invoke({"messages": messages})
        overall_task = response_org["parsed"].overall_task
        overall_task.module_description += (
            "\nProcedure:\n" + response_org["parsed"].overall_step
        )
        return {
            "task_details": response_org["parsed"].tasks,
            "overall_task": overall_task,
            "user_problem": messages[0][1],
            "messages": messages,
        }

    return organizer


def generate_task_manager(code_generator: CompiledStateGraph) -> Callable:
    def task_manager(state: GraphState) -> dict[str, list]:
        logger = getLogger(__name__).getChild("task_manager")
        logger.info("---DISTRIBUTING TASKS @ TASK MANAGER---")

        codes = []
        for i, t in enumerate(state.task_details, start=1):
            logger.info(f"---CODE GENERATING ({i}/{len(state.task_details)})---")
            res = code_generator.invoke(TaskState(task=t))
            codes.append(res["code"])

        return {"generation": codes}

    return task_manager


def obtain_finalizer_prompt_oai() -> ChatPromptTemplate:
    """
    コード生成LLM用のプロンプトを生成します。

    Returns:
        ChatPromptTemplate: LLMで使用されるプロンプト。
    """
    system_message = strip_block_comment(
        """
        You are an expert Python programmer.
        Create Python code based on the following specifications.
        Try to use all the python modules in `context` given by user.
        The modules in the `context` need to be copied to the current code.
        Structure your response as follows:

        1.  A brief description of the code.
        2.  The import statements (if any. can be empty "").
            **Do not** include function definitions in the import block.
            Type annotation syntax is based on Python 3.13.
            **Do NOT import `Dict`, `List`, `Tuple` from the `typing` module.**
            **Use built-in types `dict`, `list`, `tuple` directly in type hints.**
        3.  The functional code block.
            This code should use explicit type annotations as accurately as possible.
            Target Python version: 3.13, ensuring better code clarity and maintainability.
            Include comprehensive docstring focusing on detailed module descriptions.
            Include exectable call at the end: `if __name__ == "__main__":`.
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

        # Input examples
        {module_input_example}

        # Output examples
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


def create_finalizer_context(codes: list[Code]) -> str:
    final_code = merge_and_format_code(
        import_blocks=[c.imports for c in codes],
        code_blocks=[c.code for c in codes],
    )
    context: str = (
        "Use following modules to create the module.\n\n"
        + "## modules\n"
        + "```python\n"
        + final_code
        + "```"
    )
    return context


def generate_finalizer(code_generator: CompiledStateGraph) -> Callable:
    def finalizer(state: GraphState):
        """
        Generate final overall code
        """
        logger = getLogger(__name__).getChild("finalizer")
        logger.info("---FINILIZING CODE @ FINALIZER---")

        response_finalizer = code_generator.invoke(
            {
                "context": create_finalizer_context(state.generation),
                "task": state.overall_task,
            }
        )
        return {"generation": state.generation + [response_finalizer["code"]]}

    return finalizer


def main(question_file: Path, graph_plot: bool, interact: bool, verbose: int) -> None:
    load_dotenv()
    logger = getLogger(__name__)
    set_base_log_level(verbose)

    set_log_level(["matplotlib", "httpx"])

    with open(question_file, "r", encoding="utf-8") as file:
        question = file.read()

    # --- graph definition ---
    structured_llm_chain_org = obtain_structured_llm_oai(
        output_structure=TaskList, prompt=obtain_organizer_prompt_oai()
    )

    # === graph definition ===
    workflow = StateGraph(GraphState)

    # --- coding subgraph ---
    code_generation_app = generate_code_generation_graph()
    code_generation_app_for_finalizer = generate_code_generation_graph(
        chain=obtain_structured_llm_oai(
            output_structure=Code, prompt=obtain_finalizer_prompt_oai()
        )
    )

    # Define the nodes
    workflow.add_node("organizer", generate_organizer(structured_llm_chain_org))
    workflow.add_node("task_manager", generate_task_manager(code_generation_app))
    workflow.add_node(
        "finalizer", generate_finalizer(code_generation_app_for_finalizer)
    )

    workflow.add_edge(START, "organizer")
    workflow.add_edge("organizer", "task_manager")
    workflow.add_edge("task_manager", "finalizer")
    workflow.add_edge("finalizer", END)

    app = workflow.compile()
    app.name = "main_graph"

    if graph_plot:
        plot_graph([app, code_generation_app])

    solution = app.invoke({"messages": [("user", question)]})

    logger.info("\n===== sub-modules =====")
    logger.info("\n" + codes_to_str(solution["generation"][:-1]))
    logger.info("\n===== main function =====")
    logger.info("\n" + codes_to_str([solution["generation"][-1]]))

    if interact:
        ipdb.set_trace()


if __name__ == "__main__":
    main(**vars(_retrieve_args()))
