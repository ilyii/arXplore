# Example: reuse your existing OpenAI setup
from openai import OpenAI
import bs4
from langchain import hub
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_chroma import Chroma
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Point to the local server


COLLECTION_NAME="master"

class Embedder:
      def __init__(self, model="second-state/All-MiniLM-L6-v2-Embedding-GGUF"):
         self.model = model
         self.client = OpenAI(base_url="http://localhost:5000/v1", api_key="lm-studio")

      def embed_documents(self, input):
         return [d.embedding for d in self.client.embeddings.create(input = input, model=self.model).data]

      def embed_query(self, query: str):
            return self.client.embeddings.create(input = query, model=self.model).data[0].embedding

from langchain_openai import ChatOpenAI




def main():
    embedder = Embedder()
    vectorstore = Chroma(collection_name=COLLECTION_NAME, embedding_function=embedder, persist_directory=".", create_collection_if_not_exists=False)
    retriever = vectorstore.as_retriever()

    # 2. Incorporate the retriever into a question-answering chain.
    system_prompt = (
        "You are an assistant for question-answering tasks. "
        "Use the following pieces of retrieved context to answer "
        "the question. If you don't know the answer, say that you "
        "don't know. Use three sentences maximum and keep the "
        "answer concise."
        "\n\n"
        "{context}"
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{input}"),
        ]
    )

    llm = ChatOpenAI(base_url="http://localhost:5000/v1", api_key="lm-studio")
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)


    response = rag_chain.invoke({"input": "Give me a summary of the article."})
    print(response["answer"])

    from langchain.tools.retriever import create_retriever_tool

    tool = create_retriever_tool(
        retriever,
        "blog_post_retriever",
        "Searches and returns excerpts from the Autonomous Agents blog post.",
    )
    tools = [tool]

    print(tool.invoke("task decomposition"))


if __name__ == "__main__":
    main()