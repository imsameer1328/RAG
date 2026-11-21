from langchain_ollama import ChatOllama


def answer_question(vector_store, question):
    """
    Retrieve relevant documentation and generate an answer using Gemma.
    """

    # 1. Initialize the LLM
    llm = ChatOllama(
        model="gemma2:2b",
        temperature=0
    )

    # 2. Retrieve relevant document chunks
    retriever = vector_store.as_retriever(
        search_kwargs={"k": 3}
    )

    documents = retriever.invoke(question)

    if not documents:
        return {
            "answer": "I couldn't find relevant information in the documentation.",
            "sources": []
        }

    # 3. Prepare context and source references
    context_parts = []
    sources = []

    for index, doc in enumerate(documents, start=1):
        source = doc.metadata.get("source", "Unknown source")

        context_parts.append(
            f"[Source {index}: {source}]\n{doc.page_content}"
        )

        if source not in sources:
            sources.append(source)

    context = "\n\n".join(context_parts)

    # 4. Construct the prompt
    prompt = f"""
You are DocuMind AI, an assistant that answers questions
using technical documentation.

Rules:
- Answer using only the provided context.
- Do not invent information.
- If the context does not contain the answer, say so.
- Keep the answer clear and useful.
- Include source references such as [Source 1] where appropriate.

Documentation Context:
{context}

User Question:
{question}

Answer:
"""

    # 5. Generate the answer
    response = llm.invoke(prompt)

    return {
        "answer": response.content,
        "sources": sources
    }
