import os
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import streamlit as st
import streamlit.components.v1 as components
from chromadb import PersistentClient
from chromadb.utils import embedding_functions
from openai import OpenAI

# Constants
CHROMA_DATA_PATH = "src/chroma_data"
EMBED_MODEL = "multi-qa-MiniLM-L6-cos-v1"
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

# Define the spinning ArXiv logo HTML and CSS
loading_spinner_html = """
<div class="spinner-container">
    <div class="spinner"></div>
</div>
<style>
    .spinner-container {
        position: relative;
        width: 100%;
        height: 100%;
        display: flex;
        align-items: center;
        justify-content: center;
    }
    .spinner {
        width: 100px;
        height: 100px;
        background-image: url('https://upload.wikimedia.org/wikipedia/commons/thumb/b/bc/ArXiv_logo_2022.svg/2560px-ArXiv_logo_2022.svg.png');
        background-size: contain;
        background-repeat: no-repeat;
        animation: spin 2s linear infinite;
    }
    @keyframes spin {
        from {transform: rotate(0deg);}
        to {transform: rotate(360deg);}
    }
</style>
"""

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

# Input arXiv ID or URL
arxiv_input = st.text_input(
    "Enter arXiv ID or URL",
    help="Provide an arXiv ID or URL for a specific paper.",
    placeholder="e.g., 2105.12345 or https://arxiv.org/abs/2105.12345",
)

# Summarization toggle
summarize_toggle = st.checkbox("Generate summary using LLM")

# Input text
input_text = st.text_area(
    "Enter text for recommendations",
    height=200,
    help="Enter text to get recommendations.",
    key="input_text",
    placeholder="Embedding models for text similarity and retrieval system",
    value="Embedding models for text similarity and retrieval system",
    max_chars=1000,
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
    Highlight words in the text with green color.
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
        temperature=0.3,
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


def create_markdown(title, abstract, url, authors, year, category, summary=None):
    """
    Create markdown content with paper details.
    """
    markdown_content = f"# {title}\n\n"
    markdown_content += f"**URL:** [{url}]({url})\n\n"
    markdown_content += f"**Authors:** {', '.join(authors)}\n\n"
    markdown_content += f"**Year:** {year}\n\n"
    markdown_content += f"**Category:** {category}\n\n"
    markdown_content += f"## Abstract\n\n{abstract}\n\n"

    if summary:
        markdown_content += f"## Summary\n\n{summary}\n\n"
        markdown_content += "_(Generated by AI - please verify for accuracy)_\n\n"

    return markdown_content


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
        doc_title, doc_abstract = _doc.split("[SEP]")
        doc_category = _meta["super_category"]
        doc_date = _meta["update_date"]
        doc_authors = _meta.get("authors", ["Unknown Author"])

        st.markdown(
            f"### [{highlight_words(doc_title.strip(), input_text.split() + keywords)}](https://arxiv.org/pdf/{_id}.pdf)"
        )
        st.write(f"**Distance:** {_dist}")
        st.write(f"**Category:** {doc_category}")
        st.write(f"**Date:** {doc_date}")
        st.write(f"**Abstract:** {highlight_words(doc_abstract.strip(), input_text.split() + keywords)}")

        if summarize_toggle:
            # Show the loading spinner
            loading_placeholder = st.empty()
            loading_placeholder.markdown(loading_spinner_html, unsafe_allow_html=True)

            summary = generate_summary(doc_title, doc_abstract, doc_category, doc_date, input_text)

            # Remove the loading spinner and show the summary
            loading_placeholder.empty()
            st.markdown(
                f'<div style="font-family: Arial; background-color: #3F4A99; padding: 10px; border-radius: 5px;">'
                f"{summary}<br><br><small>(Generated by AI - please verify for accuracy)</small>"
                "</div>",
                unsafe_allow_html=True,
            )
        else:
            summary = None

        markdown_content = create_markdown(
            doc_title, doc_abstract, f"https://arxiv.org/pdf/{_id}.pdf", doc_authors, doc_date, doc_category, summary
        )
        st.download_button(
            label="Download Markdown", data=markdown_content, file_name=f"{_id}.md", mime="text/markdown"
        )

        st.write("---")


if st.button("Get Recommendations"):
    st.empty()
    if input_text:
        specific_paper_query = False
        if arxiv_input:
            arxiv_id = arxiv_input.split("/")[-1] if "arxiv.org" in arxiv_input else arxiv_input
            searched_doc = collection.get(ids=[str(arxiv_id)], include=["embeddings", "metadatas", "documents"])
            arxiv_id = format_arxiv_id(arxiv_id)
            if searched_doc["ids"]:
                s_text = searched_doc["documents"][0]
                query_results = collection.query(query_texts=[s_text], n_results=num_recommendations + 1)
                specific_paper_query = True

                st.markdown(
                    f'<div style="font-family: Arial; background-color: #3F4A99; padding: 10px; border-radius: 5px;">'
                    f"<b>Specific Paper Found:</b> {s_text.split('[SEP]')[0]} "
                    f"<a href='https://arxiv.org/pdf/{arxiv_id}.pdf' target='_blank'>[PDF]</a>"
                    "</div>",
                    unsafe_allow_html=True,
                )
            else:
                st.error(f"Paper with arXiv ID {arxiv_id} not found.")
                st.stop()
        else:
            query_results = collection.query(query_texts=[input_text], n_results=num_recommendations)

        display_results(query_results, input_text, arxiv_id if specific_paper_query else None)
    else:
        st.error("Please enter some text for recommendations.")
