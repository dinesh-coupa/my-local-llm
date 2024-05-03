from langchain_community.llms import CTransformers
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.vectorstores.chroma import Chroma
from langchain.chains import RetrievalQA

__import__("pysqlite3")
import sys

sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")


loader = TextLoader("solutions/files/stalin_fide.txt")
documents = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=250)
texts = text_splitter.split_documents(documents)

# print(texts[0])
# print(texts[1])

embeddings = HuggingFaceEmbeddings()
store = Chroma.from_documents(texts, embeddings, collection_name="stalin-fide")

llm = CTransformers(
    model="zoltanctoth/orca_mini_3B-GGUF", model_file="orca-mini-3b.q4_0.gguf"
)
chain = RetrievalQA.from_chain_type(llm, retriever=store.as_retriever())

prompt = "Who is Stalin? what is Stalin talking about ?"
print(llm(prompt))
print(chain.invoke("Who is Stalin? what is Stalin talking about ?"))

# zoltanctoth/orca_mini_3B-GGUF
# model_file="orca-mini-3b.q4_0.gguf"
