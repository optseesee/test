from logging import getLogger
import ast
import os
from io import BytesIO
from pathlib import Path
from textwrap import dedent
from typing import Type

import black
import ipdb  # noqa E402, type: ignore
import isort.api
import matplotlib.pyplot as plt
from bs4 import BeautifulSoup as Soup
from langchain_community.document_loaders.recursive_url_loader import RecursiveUrlLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langgraph.graph.state import CompiledStateGraph
from pydantic import BaseModel

from states import Code


def check_parse_error(tool_output):
    """Check for parse error or failure to call the tool"""
    logger = getLogger(__name__).getChild("check_parse_error")

    # Error with parsing
    if tool_output["parsing_error"]:
        # Report back output and parsing errors
        logger.error("Parsing error!")
        raw_output = str(tool_output["raw"].content)
        error = tool_output["parsing_error"]
        raise ValueError(
            f"Error parsing your output! Be sure to invoke the tool. Output: {raw_output}. \n Parse error: {error}"
        )

    # Tool was not invoked
    elif not tool_output["parsed"]:
        logger.error("Failed to invoke tool!")
        raise ValueError(
            "You did not use the provided tool! Be sure to invoke the tool to structure the output."
        )
    return tool_output


def obtain_structured_llm_oai(
    output_structure: Type[BaseModel],
    prompt: ChatPromptTemplate,
    expt_llm: str = "gpt-4o-mini",
    include_raw: bool = True,
):

    llm = ChatOpenAI(temperature=0, seed=42, model=expt_llm)
    code_gen_chain_oai = (
        prompt
        | llm.with_structured_output(output_structure, include_raw=include_raw)
        | check_parse_error
    )
    return code_gen_chain_oai


def obtain_doc(urls: str | list[str], max_depth: int = 20) -> str:
    if isinstance(urls, str):
        urls = [urls]
    docs = []
    for url in urls:
        loader = RecursiveUrlLoader(
            url=url,
            max_depth=max_depth,
            extractor=lambda x: Soup(x, "html.parser").text,
        )
        docs += loader.load()

    # Sort the list based on the URLs and get the text
    d_sorted = sorted(docs, key=lambda x: x.metadata["source"])
    d_reversed = list(reversed(d_sorted))
    concatenated_content = "\n\n\n --- \n\n\n".join(
        [doc.page_content for doc in d_reversed]
    )
    return concatenated_content


def _set_env(credentials: dict[str, str]) -> None:
    """
    set environmental variables given by {key: value} type dict
    """
    for key, value in credentials.items():
        os.environ[key] = value


def _load_credentials(file_paths: list[Path]) -> dict[str, str]:
    """
    read credential files formatted as follows
        CREDENTIAL_KEY=CREDENTIAL_VALUE
    """
    credentials: dict[str, str] = {}
    for p in file_paths:
        if not p.exists():
            raise FileNotFoundError(f"{p} NOT FOUND")
        with p.open("r") as file:
            for line in file:
                line = line.strip()
                if line.startswith("#"):  # skip comments
                    continue
                key, value = line.split("=", 1)
                credentials[key] = value
    return credentials


def load_and_set_credentials(file_paths: list[Path]) -> None:
    _set_env(_load_credentials(file_paths))


def strip_block_comment(comment: str) -> str:
    lines = comment.split("\n")
    non_blank_lines = [line for line in lines if line.strip() != ""]
    return dedent("\n".join(non_blank_lines))


def plot_graph(
    graph: CompiledStateGraph | list[CompiledStateGraph], out: Path | None = None
) -> None:
    plt.switch_backend(
        "TkAgg"
    )  # for error `RuntimeError: main thread is not in main loop`
    if not isinstance(graph, list):
        graph = [graph]

    n_rows = len(graph)
    fig, axes = plt.subplots(1, n_rows, figsize=(5 * n_rows, 5))

    if n_rows == 1:
        axes = [axes]
    for i, g in enumerate(graph):
        img_bin = g.get_graph().draw_mermaid_png()
        img_stream = BytesIO(img_bin)
        img = plt.imread(img_stream, format="png")
        axes[i].imshow(img)
        axes[i].axis("off")
        axes[i].set_title(f"{g.name}")

    plt.tight_layout()
    if out:
        plt.savefig(out)

    plt.show()
    plt.close(fig)


def collect_imports(import_blocks):
    """import文ブロックからimport文を収集し、重複を排除します。"""
    logger = getLogger(__name__).getChild("collect_imports")
    imports = set()
    for block in import_blocks:
        try:
            tree = ast.parse(block)
            for node in ast.walk(tree):
                if isinstance(node, (ast.Import, ast.ImportFrom)):
                    imports.add(ast.unparse(node))
        except SyntaxError as e:
            logger.error(f"importブロック構文エラー: {e}")
            return None
    return imports


def format_with_isort(code):
    """コード文字列をisortでフォーマットします。"""
    logger = getLogger(__name__).getChild("format_with_isort")
    try:
        formatted_code = isort.api.sort_code_string(code, config=isort.Config())
        return formatted_code
    except Exception as e:
        logger.error(f"isort format ERROR: {e}")
        ipdb.set_trace()
        return None


def format_with_black(code):
    """コード文字列をblackでフォーマットします。"""
    logger = getLogger(__name__).getChild("format_with_black")
    try:
        formatted_code = black.format_str(code, mode=black.FileMode())
        return formatted_code
    except black.InvalidInput as e:
        logger.error(f"black format ERROR: {e}")
        ipdb.set_trace()
        return None


def merge_and_format_code(import_blocks, code_blocks):
    """import文とコードを統合し、isortとblackでフォーマットします。"""
    merged_code = "\n".join(import_blocks) + "\n" + "\n".join(code_blocks)

    isort_formatted = format_with_isort(merged_code)
    if isort_formatted is None:
        return None

    black_formatted = format_with_black(isort_formatted)
    if black_formatted is None:
        return None

    return black_formatted


def code_to_str(code: Code) -> str:
    """与えられた Code オブジェクトを文字列に変換します。

    Args:
        code: 変換する Code オブジェクト

    Returns:
        コード文字列。
    Raises:
        TypeError: Code でない場合に発生。
    """
    if not isinstance(code, Code):
        raise TypeError("引数は Code オブジェクトにしてください。")

    return codes_to_str([code])


def codes_to_str(codes: list[Code]) -> str:
    """与えられた Code オブジェクトのリストを文字列に変換します。

    Args:
        codes: 変換する Code オブジェクトのリスト。

    Returns:
        結合されたコード文字列。

    Raises:
        TypeError: List[Code] でない場合に発生。
    """
    if not isinstance(codes, list):
        raise TypeError("引数は Code オブジェクトのリストにしてください。")
    if not all(isinstance(c, Code) for c in codes):
        raise TypeError("引数は Code オブジェクトのリストにしてください。")

    import_blocks = [c.imports for c in codes]
    code_blocks = [c.code for c in codes]
    return merge_and_format_code(import_blocks, code_blocks)
