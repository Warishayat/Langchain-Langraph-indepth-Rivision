
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter


loader = PyPDFLoader("Purposal.pdf")
docs = loader.load()
print(len(docs))

#chunk_overlap = yea batata h ap 2 chunks k between kitny chracter ka overlap hoga
# is ka faeda kia hae? context ko retain krte haen agr humry chunks m koi word
# beech se kat jaye to.mean uska context jo h wo cut hojta hae then overlap 
# came in game.
#disdvatages : zyada compution zaya hoga agr zyada use krogy 
# 10to20% is good number for overlapping

splitter = CharacterTextSplitter(chunk_size=200,c
hunk_overlap=0,separator="")
chunks = splitter.split_documents(docs)
print(len(chunks))
print(chunks[0].page_content)
print()
print("2nd Chunk")
print(chunks[1].page_content)
# Length base text splitting --->it is simple and fast process we decide
# already what will be the size of chunks etc. no grammer,no spelling.
# the biggest flaws of this text splitter it dont care about words 
# grammer etc when it will look 100 character are complete it brake the chinks
# as many time you will see it dont care wheather the sentense is complete or not

