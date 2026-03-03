import os
from pathlib import Path


from langchain_community.embeddings import DashScopeEmbeddings
from langchain_chroma import Chroma


class my_chroma_database(Chroma):
    def __init__(self):
        self.embaddings = DashScopeEmbeddings(
            model="text-embedding-v1",
            dashscope_api_key=os.getenv("DASHSCOPE_API_KEY")
        )

        current_dir = Path(__file__).parent
        self.collection_name="chinese_laws"
        self.embedding_function=self.embaddings
        self.persist_directory=str(current_dir / 'chromadb')
        self.collection_metadata={"hnsw:space" : "l2"}

        super().__init__(
            collection_name=self.collection_name,
            embedding_function = self.embedding_function,
            persist_directory = self.persist_directory,
            collection_metadata = self.collection_metadata
        )

if __name__ == "__main__":
    import os
    print(os.path.exists("data/chromadb"))
    db = my_chroma_database()
    res = db._collection
    print(res.count())
