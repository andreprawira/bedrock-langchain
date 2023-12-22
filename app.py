import warnings
warnings.filterwarnings('ignore')
import os
import sys
from utils import print_ww
import boto3
from langchain.embeddings import BedrockEmbeddings
from langchain.llms.bedrock import Bedrock
from urllib.request import urlretrieve
import numpy as np
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader, PyPDFDirectoryLoader, DirectoryLoader
from langchain.vectorstores import FAISS
from langchain.indexes.vectorstore import VectorStoreIndexWrapper
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
import streamlit as st



boto3_bedrock = boto3.client("bedrock-runtime")

# - create the Anthropic Model
llm = Bedrock(model_id="anthropic.claude-v2:1", 
              client=boto3_bedrock, 
              model_kwargs={'max_tokens_to_sample':200})
bedrock_embeddings = BedrockEmbeddings(model_id="amazon.titan-embed-text-v1", client=boto3_bedrock)

os.makedirs("data", exist_ok=True)
files = [
    "https://www.irs.gov/pub/irs-pdf/p1544.pdf",
    "https://www.irs.gov/pub/irs-pdf/p15.pdf",
    "https://www.irs.gov/pub/irs-pdf/p1212.pdf",
]
for url in files:
    file_path = os.path.join("data", url.rpartition("/")[2])
    urlretrieve(url, file_path)

# loader = PyPDFDirectoryLoader("./data/")
loader = DirectoryLoader("./data/", recursive=True, glob="*.py")

documents = loader.load()
# - in our testing Character split works better with this PDF data set
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 1000,
    chunk_overlap = 100,
)
docs = text_splitter.split_documents(documents)
    
vectorstore_faiss = FAISS.from_documents(
    docs,
    bedrock_embeddings,
)

wrapper_store_faiss = VectorStoreIndexWrapper(vectorstore=vectorstore_faiss)

prompt_template = """

Human: Use the following pieces of context to provide a concise answer to the question at the end. If you don't know the answer, simply say idk lol.
<context>
{context}
</context

Question: {question}

Assistant:"""

PROMPT = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"]
)

qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectorstore_faiss.as_retriever(
        search_type="similarity", search_kwargs={"k": 3}
    ),
    return_source_documents=True,
    chain_type_kwargs={"prompt": PROMPT}
)


# Comment out line 84 to 93 and uncomment anything below line 97 if you want to use Streamlit 

#query = "will i go to jail if i file my taxes incorrectly?"
#query = "apakah user pool region di file CognitoUserToCSV.py itu argumen yang diwajibkan? apakah ada nilai bawaan region untuk user pool? jika iya, apakah nilai bawaan untuk user pool region?"
#query = "how do i modify bedrock-workshop.py so i can load all files inside subfolders when using DirectoryLoader class? Or do i need to use different class? show me the code snippet"
#query = "is user pool region in CognitoUserToCSV.py a required argument? is there a default value for it?"
#query = "explain to me what does dynamo_db.py code do?"
#query = "how many methods are there defined in s3.py?"
#query = "whos john cena?"

# result = qa({"query": query})
# print_ww(result['result'])

st.title('🤠 Hotdawg AI')
st.write("I can help you answer questions you have based on your own knowledge base.")
st.subheader('Question:')
query = st.text_input("Whats on ur mind bruh?")
st.subheader('Answer: ')
result = qa({"query": query})
st.write(result['result']) 