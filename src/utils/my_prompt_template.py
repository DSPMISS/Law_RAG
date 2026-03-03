
from langchain.messages import SystemMessage, HumanMessage


class PrimaryPromptTemplate(list):
    def __init__(self, question):
        sys_message = SystemMessage(
        "你是一个法律助手，现在有一位客户需要询问你一些法律相关问题，请用人类的语气相对简洁的他的问题"
        )
        human_message = HumanMessage(question)
        super().__init__([sys_message, human_message])

if __name__ == '__main__':
    from model import model
    message = PrimaryPromptTemplate("有人绑架了我的女朋友，我该怎么办？")
    answer = model.qwen.invoke(message)
    print(answer.content_blocks)
