# What id docuemnt loader?
# nothing but componnets but load the data into langchain.
# document formate is somethign  lik e when we pull the document 
# it will be in the form of docuemnt there are 2 thing in it.
# 1: page_content
# 2: meta data
from langchain_community.document_loaders import TextLoader
loader = TextLoader(r"C:\Users\HP\Desktop\Langchain_Langgraph\Langchain-Langraph-indepth-Rivision\Langchain\Indexes\doc_loader\keys.txt",encoding='utf-8')
text_docs=loader.load()
print(text_docs[0])

#lets check meta data and page_Conten
print("content of the text file")
print(text_docs[0].page_content)
print()
print("Meta data of  the file")
print(text_docs[0].metadata)


