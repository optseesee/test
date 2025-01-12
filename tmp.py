from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables.base import RunnableSequence
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
import os
from dotenv import load_dotenv



from tools import (
    codes_to_str,
    merge_and_format_code,
    obtain_structured_llm_oai,
    plot_graph,
    strip_block_comment,
)
from utils import set_base_log_level, set_log_level

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
print(api_key)

def main():
    # メイン処理を書く
    print("Hello, World!")

if __name__ == "__main__":
    main()


import sys
import langchain

print(f"Pythonのバージョン：{sys.version}")
print(f"LangChainのバージョン：{langchain.__version__}")
#print(f"OpenAIのバージョン：{openai.__version__}")
