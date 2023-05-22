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