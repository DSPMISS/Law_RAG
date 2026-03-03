import os

from fsspec.asyn import private

from my_chromadb import *

import pandas as pd
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter


class data_splitter():
    def __init__(self, root_path):

        self.__has_database = 0

        self.header = header = [
        ("#", "header1"),
        ("##", "header2"),
        ("###", "header3"),
        ("####", "header4"),
        ]
        self.markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=header,
                                             return_each_line=False,
                                             strip_headers=True)
        self.recursive_splitter = RecursiveCharacterTextSplitter(
            chunk_size=200,
            chunk_overlap=50,
            separators=[
                "第*.条"
                "\n",
                ""
            ]
        )
        self.chromadb = my_chroma_database().database

        self.root_path = root_path

    def read_law(self):
        law_splited = []
        law_paths = []
        root_path = self.root_path

        for root, dirs, files in os.walk(root_path):
            if files != []:
                for file in files:
                    if file != "_index.md":
                        law_paths.append(os.path.join(root, file))
        for index, path in enumerate(law_paths):
            with open(path, "r", encoding="utf-8") as f:
                content = f.readlines()
                content = [item for item in content
                           if (item != "\n") and
                           ("<!-- INFO END -->" not in item) and
                           ("<!-- FORCE BREAK -->" not in item)]
                content = "".join(content)
                splited_content = self.markdown_splitter.split_text(content)
                splited_content = self.recursive_splitter.split_documents(splited_content)
                self.chromadb.add_documents(splited_content)
                print(f"{index}/{len(law_paths)}," + str(path))
        self.__has_database = 1

    def get_database(self):
        return self.chromadb

    def get_has_database(self):
        return self.__has_database


if __name__ == '__main__':
    spliter = data_splitter("Laws")
    if not spliter.get_has_database():
        spliter.read_law()
    database = spliter.get_database()

