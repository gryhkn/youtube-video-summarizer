import os
from langchain.document_loaders import YoutubeLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain import OpenAI
from langchain.chains.summarize import load_summarize_chain
from dotenv import load_dotenv
load_dotenv()
openai_api_key = os.environ.get("OPENAI_API_KEY")

llm = OpenAI(temperature=0)
loader = YoutubeLoader.from_youtube_url("https://www.youtube.com/watch?v=lyE5d7zLxvc", add_video_info=False, language="tr")
docs = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=1000, separator="", chunk_overlap=0)
split_docs = text_splitter.split_documents(docs)
chain = load_summarize_chain(llm, chain_type="map_reduce")
output_summary = chain.run(split_docs)
print(output_summary)



# llm = OpenAI(temperatur=0)
# embeddings = OpenAIEmbeddings()
# db = Chroma.from_documents(texts, embeddings)
#
#
#
#
#
#
# documents = loader.load()
# texts = text_splitter.split_documents(documents)
#
#
#
# #OPENAI_API_KEY = "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"
# #openai_api_key=OPENAI_API_KEY
#
#
# def extract_youtube_id(youtube_url):
#     return youtube_url.split('v=')[-1]
#
#
# def load_and_vectorize(youtube_url_id):
#     loader = YoutubeLoader.from_youtube_url(youtube_url_id, add_video_info=False, language="tr")
#     docs = loader.load()
#     index = VectorstoreIndexCreator().from_loaders(docs)
#     return index
#
#
# def query_index(index, query):
#     return index.query(query)
#
#
# def main():
#     url = input("Youtube video linki girin: ")
#
#     youtube_url_id = extract_youtube_id(url)
#     print("youtube_url_id: ", youtube_url_id)
#     index = load_and_vectorize(youtube_url_id)
#
#     query = input("Sorunuz nedir? (çıkmak için 'quit' yazın)")
#
#     while True:
#         response = query_index( index, query)
#         print(f"Answer: {response}")
#         query = input("Sorunuz nedir? (çıkmak için 'quit' yazın)")
#         if query == "quit" or query == "q":
#             break
#
#
# if __name__ == '__main__':
#     main()