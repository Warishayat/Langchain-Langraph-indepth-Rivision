from langchain_community.document_loaders import PyPDFLoader

# it is internally use two libraries like request and beautiful_soap
# static pages k sath zayada acha kam karta
# if we want to load multiple pdf then we will load directoryLoader
loader = PyPDFLoader(r"C:\Users\HP\Desktop\Langchain_Langgraph\Langchain-Langraph-indepth-Rivision\Langchain\Indexes\doc_loader\Purposal.pdf")
docs = loader.lazy_load()
print(len(docs))

print()
print(docs)
print("This is the page content")
print(docs[2].page_content)

print()
print("This is the meta data of the pdf")
print(docs[2].metadata)


# there are 4 pages in the data and you will get the 4
# douments in return and you can extract that one nby oen 
# and all combine.


#load function mean eager loading 500 pages ko ik sth memory m load krega
# lazy_loading alg hae is m age 500 pages hngy tu return m genrator deta h return m
# use lazy_load when the pdf is big.