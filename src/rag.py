import os
import re
import time
from pathlib import Path
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from vectorstore import load_vectorstore

load_dotenv()

PROMPT = ChatPromptTemplate.from_template(
    """You are a research assistant helping a student understand research papers.
Answer the question using ONLY the context below.

Write a clear, detailed answer of 2 short paragraphs (about 120 to 180 words):
- First paragraph: a simple definition in easy language.
- Second paragraph: how it works or why it matters, using details from the context.
Cite the sources you use as [1], [2], etc. after the sentences they support.

If the context does not contain the answer, say exactly:
"I could not find this in the provided papers."
Do not add facts that are not in the context.

Question: {question}

Context:
{context}

Answer:"""
)


def get_llm():
    return ChatGoogleGenerativeAI(
        model=os.getenv("GEMINI_MODEL", "gemini-3.6-flash"),
        google_api_key=os.getenv("GOOGLE_API_KEY"),
       
    )


def format_context(docs):
    parts = []
    for i, d in enumerate(docs, 1):
        name = Path(d.metadata["source"]).name
        page = d.metadata["page"] + 1
        parts.append(f"[{i}] ({name}, page {page})\n{d.page_content}")
    return "\n\n".join(parts)


def ask_with_retry(chain, inputs, tries=4, wait=10):
    for attempt in range(1, tries + 1):
        try:
            return chain.invoke(inputs)
        except Exception as e:
            temporary = any(code in str(e) for code in ("503", "429", "UNAVAILABLE"))
            if attempt == tries or not temporary:
                raise
            print(f"Attempt {attempt} failed. Retrying in {wait}s...")
            time.sleep(wait)
            wait *= 2


def answer_question(question, k=8):
    db = load_vectorstore()
    docs = db.similarity_search(question, k=k)
    chain = PROMPT | get_llm() | StrOutputParser()
    answer = ask_with_retry(
        chain, {"context": format_context(docs), "question": question}
    )
    return answer, docs
def extract_citations(answer):
    numbers = set()
    for group in re.findall(r"\[([\d,\s]+)\]", answer):
        for n in group.split(","):
            if n.strip().isdigit():
                numbers.add(int(n))
    return sorted(numbers)

if __name__ == "__main__":
    question = "How does corrective RAG evaluate retrieved documents?"
    answer, docs = answer_question(question)

    print("\nQuestion:", question)
    print("\nAnswer:\n", answer)

    cited = sorted({int(n) for n in re.findall(r"\[(\d+)\]", answer)})

    print("\nSources cited:")
    for n in cited:
        if 1 <= n <= len(docs):
            d = docs[n - 1]
            print(f"[{n}] {Path(d.metadata['source']).name}, page {d.metadata['page'] + 1}")