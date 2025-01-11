import argparse
import tempfile
from logging import getLogger
from pathlib import Path
from typing import Type

import ipdb  # noqa
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel

import tools  # noqa
from states import Code
from tools import code_to_str, strip_block_comment
from utils import set_base_log_level, set_log_level


def _retrieve_args():
    """
    このツールの実行時標準入力のハンドリングのための関数。
    python bare_llm.py -h
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
        help="not plot for LangGraph by this option",
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

    return ChatPromptTemplate.from_messages(
        [
            ("system", system_message),
            ("placeholder", "{messages}"),
        ]
    )


def obtain_structured_llm_oai_without_parsecheck(
    output_structure: Type[BaseModel],
    prompt: ChatPromptTemplate,
    expt_llm: str = "gpt-4o-mini",
    include_raw: bool = False,
):
    llm = ChatOpenAI(temperature=0, seed=42, model=expt_llm)
    code_gen_chain_oai = prompt | llm.with_structured_output(
        output_structure, include_raw=include_raw
    )
    return code_gen_chain_oai


def main(question_file: Path, graph_plot: bool, interact: bool, verbose: int) -> None:
    load_dotenv()

    logger = getLogger(__name__)
    set_base_log_level(verbose)

    set_log_level(["matplotlib", "httpx"])

    with open(question_file, "r", encoding="utf-8") as file:
        question = file.read()

    structured_llm_chain_code = obtain_structured_llm_oai_without_parsecheck(
        output_structure=Code, prompt=obtain_code_gen_prompt_oai()
    )

    response_code = structured_llm_chain_code.invoke({"messages": [("user", question)]})

    logger.info("\n===== main function =====")
    logger.info("\n" + code_to_str(response_code))

    if interact:
        ipdb.set_trace()


if __name__ == "__main__":
    main(**vars(_retrieve_args()))
