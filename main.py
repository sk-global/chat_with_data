import os
import streamlit as st
from llama_index import SimpleDirectoryReader
from llama_index import LLMPredictor, GPTVectorStoreIndex, PromptHelper, ServiceContext
from langchain import OpenAI

OPENAI_API_KEY = os.environ['OPENAI_API_KEY']

document_path = './data/'

if 'response' not in st.session_state:
  st.session_state.response = ''


def send_click():
  query_engine = vector_index.as_query_engine(service_context=service_context,
                                       verbose=True,
                                       response_mode="compact")
  st.session_state.response = query_engine.query(st.session_state.prompt)


vector_index = None
st.title("Document Chatbot")

sidebar_placeholder = st.sidebar.container()
uploaded_file = st.file_uploader("Choose a pdf file")

if uploaded_file is not None:

  document_files = os.listdir(document_path)
  for doc_file in document_files:
    os.remove(document_path + doc_file)

  bytes_data = uploaded_file.read()
  with open(f"{document_path}{uploaded_file.name}", 'wb') as f:
    f.write(bytes_data)

  loader = SimpleDirectoryReader(document_path, recursive=True, exclude_hidden=True)
  documents = loader.load_data()
  sidebar_placeholder.header('Current Processing Document:')
  sidebar_placeholder.subheader(uploaded_file.name)
  sidebar_placeholder.write(documents[0].get_text()[:5000] + '...')

  llm_predictor = LLMPredictor(
    llm=OpenAI(temperature=0, model_name="text-davinci-003"))

  max_input_size = 4096
  num_output = 256
  max_chunk_overlap = 0.2
  prompt_helper = PromptHelper(max_input_size, num_output, max_chunk_overlap)

  service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor,
                                                 prompt_helper=prompt_helper)

  vector_index = GPTVectorStoreIndex.from_documents(documents,
                                             service_context=service_context,
                                             prompt_helper=prompt_helper)



if vector_index != None:
  st.text_input("Ask something on uploaded document and press the below button: ", key='prompt')
  st.button("find_answer", on_click=send_click)
  if st.session_state.response:
    st.subheader("Response: ")
    st.success(st.session_state.response, icon="🤖")
