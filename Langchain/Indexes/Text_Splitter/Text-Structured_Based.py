
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter


loader = PyPDFLoader("Purposal.pdf")
docs = loader.load()
print(len(docs))



splitter = RecursiveCharacterTextSplitter(chunk_size=200)
docs2 = splitter.split_documents(docs)
print(docs2[0].page_content)



# everytext has some formate 
# text organise in prargraph
# paragraph orgaize in sentense 
# and sentense organize in words
# RecursiveCharacterTextSplitter


# RecursiveCharacterTextSplitter Start chunking based on paragraph
# if not pargraph then it start chunkig based on sentenses.
# if not sentenses then it start chunking based on the words
# if not words it start chuking based on the chracter.