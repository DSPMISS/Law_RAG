from operator import itemgetter

from langchain_core.runnables import chain, RunnableParallel, RunnableLambda

from model import model
from src.utils import Query
from utils import Query, Mid_Answer, Answer
from data import my_chromadb

qwen = model.qwen
pro_qwen = model.lora_qwen
db = my_chromadb.my_chroma_database()

query = Query()
answer = Answer()

@chain
def combine_query_and_prim_answer(com_input:dict):
    #将用户问题和初步答案合并未一个str串，作为database检索的输入
    ques = com_input["question"]
    primary_answer = com_input["primary_answer"]
    search_input = "\n".join(["问题:" + ques, "回答:" + primary_answer])
    return search_input

@chain
def db_search(input:str):
    #检索
    laws = db.similarity_search(input)
    mid = Mid_Answer(input, laws)
    return mid.return_str()


chain = (
        RunnableLambda(query.init_query) |
        RunnableParallel(
            question=RunnableLambda(lambda x: x.get_str_quary()),
            primary_answer=RunnableLambda(lambda x: x.get_primary_answer_from_qwen())
        ) |
        combine_query_and_prim_answer |
        db_search |
        RunnableLambda(answer.generate_answer)
)

if __name__ == "__main__":
    question_input = "在某家公司中，一名员工对女同事实施了性骚扰行为，女同事向公司进行举报，但公司却没有采取必要的措施来制止这种行为。\n\n公司未采取必要措施预防和制止性骚扰，导致女同事的权益受到侵害，该公司是否需要承担责任？"
    ans = chain.invoke(question_input)
    print(ans)




    # answer1 = get_primary_answer_from_qwen(ques)
    # print(answer1)
    # print("-" * 50)
    # results = db.similarity_search_with_score(question + answer1)
    # print(results)
    # print("-" * 50)
    # for doc, score in results:
    #     print(doc.page_content + f"\nscore: {score}")
    # answer = chain.invoke(input)
    # res = db._collection
    # print(res.count())
