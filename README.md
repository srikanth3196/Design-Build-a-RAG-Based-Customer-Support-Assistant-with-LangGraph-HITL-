# Design-Build-a-RAG-Based-Customer-Support-Assistant-with-LangGraph-HITL-
code 
!pip install langchain langchain-community chromadb pypdf langgraph groq sentence-transformers
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate

from langgraph.graph import StateGraph, END
from typing import TypedDict

from groq import Groq
import os
os.environ["GROQ_API_KEY"] = "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"
client = Groq(api_key=os.environ["GROQ_API_KEY"])
from google.colab import files

uploaded = files.upload()

for fn in uploaded.keys():
  print(f'User uploaded file "{fn}" with length {len(uploaded[fn])} bytes')
  # Optionally, you can save the uploaded file to a specific path
  # with open(f'/content/{fn}', 'wb') as f:
  #   f.write(uploaded[fn])
  splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=150
)
chunks = splitter.split_documents(documents)

embedding_model = HuggingFaceEmbeddings()

vectordb = Chroma.from_documents(
    documents=chunks,
    embedding=embedding_model,
    persist_directory="./chroma_db"
)
prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="""
You are a helpful customer support assistant.
Use ONLY the context below.

Context:
{context}

Question:
{question}

If unsure, say you don't know.
"""
)
class GraphState(TypedDict):
    query: str
    context: str
    scores: list
    answer: str
    confidence: float
    escalate: bool
    def retrieve(state):
    query = state["query"]

    docs_and_scores = vectordb.similarity_search_with_score(query, k=3)

    docs = []
    scores = []

    for doc, score in docs_and_scores:
        docs.append(doc.page_content)
        scores.append(score)

    context = " ".join(docs)

    return {
        "context": context,
        "scores": scores
    }
    def generate(state):
    context = state["context"]
    query = state["query"]
    scores = state.get("scores", [])

    final_prompt = prompt.format(context=context, question=query)

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant", # Updated to a currently supported Groq model
        messages=[{"role": "user", "content": final_prompt}]
    )

    answer = response.choices[0].message.content

    if scores:
        avg_distance = sum(scores) / len(scores)

        # Convert distance → similarity confidence
        similarity_conf = 1 / (1 + avg_distance)

        # Coverage signal (how much answer uses context)
        context_len = len(context.split())
        answer_len = len(answer.split())

        coverage = answer_len / (context_len + 1)

        confidence = 0.7 * similarity_conf + 0.3 * coverage
        confidence = min(confidence, 1.0)
    else:
        confidence = 0.0

    print(f"\nDEBUG → Scores: {scores}")
    print(f"DEBUG → Confidence: {confidence:.2f}")

    return {
        "answer": answer,
        "confidence": confidence
    }
    def decide(state):
    confidence = state["confidence"]
    context = state["context"]

    if not context.strip():
        return {"escalate": True}

    if confidence < 0.6:
        return {"escalate": True}

    return {"escalate": False}
    def human_escalation(state):
    print("\n Escalating to human agent...")
    human_response = input("Human response: ")
    return {"answer": human_response}

workflow = StateGraph(GraphState)

workflow.add_node("retrieve", retrieve)
workflow.add_node("generate", generate)
workflow.add_node("decide", decide)
workflow.add_node("human", human_escalation)

workflow.set_entry_point("retrieve")

workflow.add_edge("retrieve", "generate")
workflow.add_edge("generate", "decide")

workflow.add_conditional_edges(
    "decide",
    lambda s: "human" if s["escalate"] else END
)

workflow.add_edge("human", END)

app = workflow.compile()
query = input("Enter your query: ")
result = app.invoke({"query": query})

print("\n✅ Final Answer:")
print(result["answer"])

Enter your query: Can I get a refund after 6 months?

DEBUG → Scores: [1.7713977098464966, 1.7713977098464966, 1.8059463500976562]
DEBUG → Confidence: 0.29

 Escalating to human agent...
Human response: Refunds are only allowed within 30 days. Please contact support for exceptions

✅ Final Answer:
Refunds are only allowed within 30 days. Please contact support for exceptions
