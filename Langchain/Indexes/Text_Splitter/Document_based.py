from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter, Language

loader = TextLoader("text.txt", encoding="utf-8")
docs = loader.load()
print(len(docs))


splitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.MARKDOWN,
    chunk_size=100,
    chunk_overlap=0
)
docs2 = splitter.split_documents(docs)

for docs in docs2:
    print("Print docs",docs)
    print(docs.page_content)
