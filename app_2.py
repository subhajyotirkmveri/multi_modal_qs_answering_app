from langchain.document_loaders import PyPDFLoader, DirectoryLoader
import tempfile
import streamlit as st
from base_1 import retrieval_qa_chain, create_vector_db
import base64           # to read the pdf
import yaml

def pdf_loader(tmp_file_path):
    loader = PyPDFLoader(file_path=tmp_file_path)
    return loader
    
@st.cache_data
def displayPDF(file_path):
    with open(file_path, "rb") as f:
        base64_pdf = base64.b64encode(f.read()).decode('utf-8')
    pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="600" type="application/pdf"></iframe>'
    return pdf_display
    
def main():

    st.set_page_config(page_title='QA Chatbot', layout="wide", page_icon="📃", initial_sidebar_state="expanded")
    st.title("Q&A Chatbot with PDF Support 📃")

    with st.sidebar:
        st.title("Settings")
        st.markdown('---')
        
        # Configuration File Generator
        st.title("Configuration File Generator")

        # Define options for parameters
        options = {
            "RETURN_SOURCE_DOCUMENTS": [True, False],
            "VECTOR_COUNT": [1, 2, 3],
            "CHUNK_SIZE": list(range(50, 1001)),  # Extend chunk size range from 50 to 1000
            "CHUNK_OVERLAP": list(range(0, 51)),   # Extend chunk overlap range from 0 to 50
            "DB_FAISS_PATH": "db_faiss/",
            "MODEL_TYPE": ["mistral", "llama"],
            "MODEL_BIN_PATH": ["models/mistral-7b-instruct-v0.1.Q5_K_M.gguf", 'models/llama-2-7b-chat.ggmlv3.q8_0.bin', "models/Mistral-7B-Instruct-v0.1-GGUF/tree/main", "models/Mistral-7B-Instruct-v0.2-GGUF/tree/main"],
            "EMBEDDINGS": ["sentence-transformers/all-mpnet-base-v2", "sentence-transformers/all-MiniLM-L6-v2"],
            "MAX_NEW_TOKENS": [512, 1024, 2048],
            "TEMPERATURE": [round(i * 0.01, 2) for i in range(0, 101)]  # Extend temperature range from 0.00 to 1.00
        }

        # Initialize an empty dictionary to store selected parameters
        config = {}

        # Generate UI for each parameter
        for key, value in options.items():
            if isinstance(value, list):
                config[key] = st.selectbox(f"{key}:", value)
            else:
                config[key] = st.text_input(f"{key}:", value)

        # Save the configuration to a YAML file
        if st.button("Save Configuration"):
            with open("config.yml", "w") as f:
                yaml.dump(config, f)
            st.success("Configuration saved successfully!")
        st.subheader('Upload Your PDF File')
        doc = st.file_uploader("Upload your PDF file and Click Process", 'pdf')

        if st.button("Process"):
            with st.spinner("Processing"):
                if doc is not None:
                    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
                        tmp_file.write(doc.read())  # Use read() instead of getvalue()
                        tmp_file_path = tmp_file.name
            
                    st.success(f'File {doc.name} is successfully saved!')
                    
                    load = pdf_loader(tmp_file_path)
                    create_vector_db(load)
                    st.success("Process Done")
                    
                    st.markdown(displayPDF(tmp_file_path), unsafe_allow_html=True)
                else:
                    st.error("❗️Please Upload Your File❗️")

    if "messages" not in st.session_state:
        st.session_state.messages = []
        
    for message in st.session_state.messages:
        if message["role"] == "user":
            st.markdown(f"**User:** {message['content']}")
        elif message["role"] == "assistant":
            st.markdown(f"**Assistant:** {message['content']}")

    query = st.text_input("Ask the Question")
    if st.button("Submit"):
        if query:
            chain = retrieval_qa_chain()
            result = chain(query)

            output = result.get("result") if result else None
            if output:
                st.write("Result:", output)
            st.session_state.messages.append({"role": "user", "content": query})
            st.session_state.messages.append({"role": "assistant", "content": output})

if __name__ == '__main__':
    main()

