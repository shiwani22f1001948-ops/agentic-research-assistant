from typing import TypedDict, List
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import StateGraph, START, END

from vectorstore import load_vectorstore
from rag import PROMPT, get_llm, format_context, ask_with_retry, extract_citations
from pathlib import Path


class GraphState(TypedDict):
    question: str
    docs: List[Document]
    answer: str


db = load_vectorstore()


def retrieve(state: GraphState):
    docs = db.similarity_search(state["question"], k=8)
    return {"docs": docs}


def generate(state: GraphState):
    chain = PROMPT | get_llm() | StrOutputParser()
    answer = ask_with_retry(
        chain,
        {"context": format_context(state["docs"]), "question": state["question"]},
    )
    return {"answer": answer}


graph = StateGraph(GraphState)
graph.add_node("retrieve", retrieve)
graph.add_node("generate", generate)
graph.add_edge(START, "retrieve")
graph.add_edge("retrieve", "generate")
graph.add_edge("generate", END)

app = graph.compile()


if __name__ == "__main__":
    result = app.invoke({"question": "What is self-attention?"})

    print("\nAnswer:\n", result["answer"])
    print("\nSources cited:")
    for n in extract_citations(result["answer"]):
        if 1 <= n <= len(result["docs"]):
            d = result["docs"][n - 1]
            print(f"[{n}] {Path(d.metadata['source']).name}, page {d.metadata['page'] + 1}")