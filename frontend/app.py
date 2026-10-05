import os

import requests
import streamlit as st


BACKEND_URL = os.getenv(
    "BACKEND_URL",
    "http://localhost:8000"
)


st.set_page_config(
    page_title="RAG Application",
    page_icon="📚"
)


st.title("📚 RAG Application")

st.write(
    "Ask questions about the documents in the knowledge base."
)


st.sidebar.header("Document Management")

if st.sidebar.button("Re-index Documents"):
    try:
        response = requests.post(
            f"{BACKEND_URL}/ingest",
            timeout=120
        )

        response.raise_for_status()

        data = response.json()

        st.sidebar.success(
            f"Indexed {data['documents_ingested']} documents."
        )

    except requests.RequestException as error:
        st.sidebar.error(
            f"Unable to contact backend: {error}"
        )


if st.sidebar.button("Check Health"):
    try:
        response = requests.get(
            f"{BACKEND_URL}/health",
            timeout=5
        )

        response.raise_for_status()

        health = response.json()

        st.sidebar.json(health)

    except requests.RequestException as error:
        st.sidebar.error(
            f"Unable to contact backend: {error}"
        )



question = st.text_input(
    "Ask a question:",
    placeholder="How does FastAPI validate incoming data?"
)


if st.button("Ask"):
    if not question.strip():
        st.warning("Please enter a question.")

    else:
        try:
            response = requests.post(
                f"{BACKEND_URL}/ask",
                json={
                    "question": question
                },
                timeout=120
            )

            response.raise_for_status()

            data = response.json()

            st.subheader("Answer")

            st.write(data["answer"])

            st.subheader("Sources")

            if data["sources"]:
                for source in data["sources"]:
                    st.write(f"- {source}")
            else:
                st.write("No sources found.")

            st.write(
                f"Confidence: {data['confidence']}"
            )

        except requests.RequestException as error:
            st.error(
                f"Unable to contact backend: {error}"
            )