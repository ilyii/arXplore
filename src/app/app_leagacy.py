import os
import re
from io import BytesIO

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import plotly.graph_objs as go
import streamlit as st
import streamlit.components.v1 as components
from chromadb import PersistentClient
from chromadb.utils import embedding_functions
from openai import OpenAI

# Constants
DF_PATH = "data/arxiv_metadata_app_data.parquet.gzip"
CHROMA_DATA_PATH = "src/chroma_data"
EMBED_MODEL = "all-MiniLM-L12-v2"
COLLECTION_NAME = "arxiv_papers"
CACHE_DIR = "src/cache"

# Set paths
CHROMA_DATA_PATH = os.path.join(CHROMA_DATA_PATH, EMBED_MODEL)

# Initialize ChromaDB client
client = PersistentClient(path=CHROMA_DATA_PATH)
embedding_func = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name=EMBED_MODEL, device="cuda", cache_folder=CACHE_DIR
)
collection = client.get_collection(name=COLLECTION_NAME, embedding_function=embedding_func)

# Language Model setup
system_prompt = """
You are a knowledgeable assistant specialized in summarizing arXiv papers. Your task is to provide a clear and concise summary of the paper based on its title and abstract.
Please ensure the summary is friendly, informative, and easy to understand. Highlight the main contributions, methods, and findings of the paper.
You can include additional useful information if it helps to understand the paper better.
Do not mention that you are a chatbot or that this text is AI-generated. Do also not mention that this is a summary. Only output the final summary.
Print with bullet points the top 3 key contributions or findings of the paper.
Present the information naturally and professionally as if you were a human expert.

Example:
Summary:
[Summary of the paper...]

Key Contributions:
- [Contribution 1]\n
- [Contribution 2]\n
- [Contribution 3]\n

Additional Information:
[Additional information...]

Related to the user's interest:
[Additional information related to the user's interest...]
"""
user_prompt = """
A user is interested in the following topic: {}
Here is a research paper published on {} in the category {}.
Title: {}
Abstract: {}

Please provide a concise summary highlighting the key contributions and findings of this paper, relevant to the user's interest.
"""
llm = OpenAI(base_url="http://localhost:5000/v1", api_key="lm-studio")

# Streamlit App
st.title("ArXiv Paper Recommender")

# Input text
input_text = st.text_area(
    "Enter text for recommendations",
    height=200,
    help="Enter text to get recommendations. You can also search for a specific paper by entering 'id: <arXiv ID>'.",
    key="input_text",
    placeholder="Embedding models for text similarity and retrieval system",
    value="Embedding models for text similarity and retrieval system",
    max_chars=500,
)

# Recommendation settings
num_recommendations = st.slider("Number of recommendations", min_value=1, max_value=10, value=3)

# Input keywords for highlighting
keywords = st.text_input(
    "Enter keywords to highlight in the abstracts (comma-separated)",
    help="Highlight specific keywords in the abstracts.",
    key="keywords",
    placeholder="e.g., neural networks, deep learning",
)
if len(keywords) > 0 and keywords != "":
    if "," in keywords:
        keywords = [kw.strip() for kw in keywords.split(",")]
    else:
        keywords = [keywords]
else:
    keywords = []


def highlight_words(text, words):
    """
    Highlight words in the text with blue color.
    """
    for word in words:
        text = re.sub(rf"\b{word}\b", f":green[{word}]", text, flags=re.IGNORECASE)
    return text


def generate_summary(title, abstract, category, doc_date, user_input):
    """
    Generate a summary using the language model.
    """
    prompt = user_prompt.format(user_input, doc_date, category, title, abstract)
    completion = llm.chat.completions.create(
        model="lmstudio-ai/gemma-2b-it-GGUF",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ],
        temperature=0.5,
    )
    return completion.choices[0].message.content


def format_arxiv_id(arxiv_id):
    """
    Format arxiv id to have a consistent length of dddd:dddd.
    """
    try:
        parts = arxiv_id.split(".")
        return f"{parts[0]:0>4}.{parts[1]:0>4}"
    except Exception:
        return arxiv_id


def get_random_paper():
    """
    Retrieve a random paper from the collection.
    """
    collection_ids = pd.read_parquet(DF_PATH, engine="pyarrow")["id"].tolist()
    random_id = np.random.choice(collection_ids, 1, replace=False)[0]
    document = collection.get(ids=[random_id])
    return document


def display_results(query_results, input_text, arxiv_id=None):
    """
    Display the query results in the Streamlit app.
    """
    for _id, _doc, _dist, _meta in zip(
        query_results["ids"][0],
        query_results["documents"][0],
        query_results["distances"][0],
        query_results["metadatas"][0],
    ):
        if arxiv_id and _id == arxiv_id:
            continue
        _id = format_arxiv_id(_id)
        doc_title, doc_abstract = _doc.split(" [SEP] ")
        doc_category = _meta["super_category"]
        doc_date = _meta["update_date"]

        st.markdown(
            f"### [{highlight_words(doc_title.strip(), input_text.split() + keywords)}](https://arxiv.org/pdf/{_id}.pdf)"
        )
        st.write(f"**Distance:** {_dist}")
        st.write(f"**Category:** {doc_category}")
        st.write(f"**Date:** {doc_date}")
        st.write(f"**Abstract:** {highlight_words(doc_abstract.strip(), input_text.split() + keywords)}")
        st.markdown(
            f'<div style="font-family: Arial; background-color: #3F4A99; padding: 10px; border-radius: 5px;">'
            f"{generate_summary(doc_title, doc_abstract, doc_category, doc_date, input_text)}"
            "</div>",
            unsafe_allow_html=True,
        )
        st.write("---")


def display_random_paper_with_recommendations(num_recommendations):
    """
    Display a random paper along with recommendations based on it.
    """
    random_doc = get_random_paper()
    _id = format_arxiv_id(random_doc["ids"][0])
    doc_title, doc_abstract = random_doc["documents"][0].split(" [SEP] ")
    doc_category = random_doc["metadatas"][0]["super_category"]
    doc_date = random_doc["metadatas"][0]["update_date"]

    st.markdown(f"### [{doc_title.strip()}](https://arxiv.org/pdf/{_id}.pdf)")
    st.write(f"**Category:** {doc_category}")
    st.write(f"**Date:** {doc_date}")
    st.write(f"**Abstract:** {doc_abstract.strip()}")
    st.markdown(
        f'<div style="font-family: Arial; background-color: #3F4A99; padding: 10px; border-radius: 5px;">'
        f"{generate_summary(doc_title, doc_abstract, doc_category, doc_date, doc_title)}"
        "</div>",
        unsafe_allow_html=True,
    )
    st.write("---")

    query_results = collection.query(query_texts=[f"{doc_title} {doc_abstract}"], n_results=num_recommendations + 1)
    display_results(query_results, "", arxiv_id=_id)

    # Display related topics
    related_topics = ", ".join({meta["super_category"] for meta in query_results["metadatas"][0]})
    st.write(f"**Related Topics:** {related_topics}")


def display_graph(query_results):
    """
    Display a graphical representation of the recommendations.
    """
    # Create a graph
    graph = nx.Graph()

    # Add nodes and edges
    for _id, _doc, _dist, _meta in zip(
        query_results["ids"][0],
        query_results["documents"][0],
        query_results["distances"][0],
        query_results["metadatas"][0],
    ):
        doc_title, _ = _doc.split(" [SEP] ")
        formatted_id = format_arxiv_id(_id)
        graph.add_node(formatted_id, title=doc_title.strip(), url=f"https://arxiv.org/pdf/{formatted_id}.pdf")

    for _id, _dist in zip(query_results["ids"][0][1:], query_results["distances"][0][1:]):
        graph.add_edge(format_arxiv_id(query_results["ids"][0][0]), format_arxiv_id(_id), weight=_dist * 100)

    # Get positions for all nodes
    pos = nx.spring_layout(graph)

    # Create edge trace
    edge_x = []
    edge_y = []
    for edge in graph.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x += (x0, x1, None)
        edge_y += (y0, y1, None)

    edge_trace = go.Scatter(x=edge_x, y=edge_y, line=dict(width=0.5, color="#888"), hoverinfo="none", mode="lines")

    # Create node trace
    node_x = []
    node_y = []
    node_text = []
    for node in graph.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        node_text.append(
            f"{graph.nodes[node]['title']}<br><a href='{graph.nodes[node]['url']}' target='_blank'>[PDF]</a>"
        )

    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        text=node_text,
        mode="markers+text",
        hoverinfo="text",
        marker=dict(
            showscale=True,
            colorscale="YlGnBu",
            size=10,
            colorbar=dict(thickness=15, title="Node Connections", xanchor="left", titleside="right"),
            line=dict(width=2),
        ),
    )

    fig = go.Figure(
        data=[edge_trace, node_trace],
        layout=go.Layout(
            title="Recommendation Graph",
            titlefont_size=16,
            showlegend=False,
            hovermode="closest",
            margin=dict(b=20, l=5, r=5, t=40),
            annotations=[dict(text="", showarrow=False, xref="paper", yref="paper")],
            xaxis=dict(showgrid=False, zeroline=False),
            yaxis=dict(showgrid=False, zeroline=False),
        ),
    )

    st.plotly_chart(fig, use_container_width=True)


if st.button("Get Recommendations"):
    if input_text:
        specific_paper_query = False
        if "id:" in input_text:
            arxiv_id_match = re.search(r"id:\s*(\d+\.\d+)", input_text)
            if arxiv_id_match:
                arxiv_id = arxiv_id_match.group(1)
                input_text = input_text.replace(f"id: {arxiv_id}", "").strip()
                searched_doc = collection.get(ids=[str(arxiv_id)], include=["embeddings", "metadatas", "documents"])
                if searched_doc["ids"]:
                    s_text = searched_doc["documents"][0]
                    query_results = collection.query(query_texts=[s_text], n_results=num_recommendations + 1)
                    specific_paper_query = True

                    st.markdown(
                        f'<div style="font-family: Arial; background-color: #3F4A99; padding: 10px; border-radius: 5px;">'
                        f"<b>Specific Paper Found:</b> {s_text.split(' [SEP] ')[0]} "
                        f"<a href='https://arxiv.org/pdf/{arxiv_id}.pdf' target='_blank'>[PDF]</a>"
                        "</div>",
                        unsafe_allow_html=True,
                    )
                else:
                    st.error(f"Paper with arXiv ID {arxiv_id} not found.")
                    st.stop()
            else:
                st.error("Please enter a valid arXiv ID.")
                st.stop()
        else:
            query_results = collection.query(query_texts=[input_text], n_results=num_recommendations)

        display_results(query_results, input_text, arxiv_id if specific_paper_query else None)

        # Graphical representation of recommendations
        st.write("### Graphical Representation of Recommendations")
        display_graph(query_results)

        # PDF download of summaries
        # pdf_output = generate_pdf_summary(query_results, input_text)
        # st.download_button(
        #     label="Download Summaries as PDF",
        #     data=pdf_output.getvalue(),
        #     file_name="summaries.pdf",
        #     mime="application/pdf",
        # )
    else:
        st.error("Please enter some text for recommendations.")

    # Button for getting a random paper with recommendations
    if st.button("Get Random Paper with Recommendations"):
        display_random_paper_with_recommendations(num_recommendations)
