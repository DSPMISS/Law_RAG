import re

from transformers import pipeline

from model import model

qwen = model.qwen

class Query(dict):
    #对用户输入的数据进行初步处理
    def __init__(self, question = ""):
        super().__init__({"question": question})
        self.str_query = question

    def init_query(self, question):
        self.str_query = question
        return self

    def get_str_quary(self):
        return self.str_query

    def get_primary_answer_from_qwen(self, return_origin_answer=False):
        """将用户问题输入未训练的模型，得到初步答案，不涉及具体法律条款

        return_origin_answer:是否需要完整的content_blocks，默认仅返回模型回答文本
        """
        try:
            from . import PrimaryPromptTemplate
        except:
            from my_prompt_template import PrimaryPromptTemplate

        message = PrimaryPromptTemplate(self.str_query)
        qwen_answer = qwen.invoke(message).content_blocks
        if not return_origin_answer:
            return qwen_answer[0]["text"]
        else:
            return qwen_answer

class Mid_Answer:
    """
    输出：
    请根据以下相关法律条款回答客户的法律问题：
    法律条款：
        第一条 xxxxxx
        第二条 xxxxxx
    客户问题：
        在某家公司中，一名员工对女同事实施了性骚扰行为，女同事向公司进行举报，但公司却没有采取必要的措施来制止这种行为。
    """

    def __init__(self, combined_query_and_prim_answer:str, laws:list):
        self.combined_query_and_prim_answer = combined_query_and_prim_answer
        self.laws_input = laws
        self.query = self._get_query()
        self.laws = self._laws_process()
        self.__return_str = """法律条款：\n"""
        self.__return_str += self.laws
        self.__return_str += "\n客户问题:\n"
        self.__return_str += self.query

    def __str__(self):
        return self.__return_str


    def _get_query(self):
        query = self.combined_query_and_prim_answer.split("问题:")[1].split("回答:")[0].strip()
        return query

    def _laws_process(self):
        laws_list = ["" for _ in range(len(self.laws_input))]
        for idx, law in enumerate(self.laws_input):
            i = 1
            while 1:
                key = f"header{i}"
                if key in law.metadata:
                    laws_list[idx] += law.metadata[key]
                    laws_list[idx] += "\t"
                    i += 1
                else:
                    break
            law_content = self._extract_from_article(law.page_content)
            laws_list[idx] += law_content
        laws_str = "\n\n".join(laws_list)
        return laws_str

    def _extract_from_article(self, text):
        # 找到第一个"第...条 "的位置
        pattern = r"第[一二三四五六七八九十百千万零]+条 "
        match = re.search(pattern, text)

        if match:
            # 返回从第一个"第...条 "开始的内容
            return text[match.start():]
        else:
            return text

    def return_str(self):
        return self.__return_str

class Answer:
    def __init__(self):
        self.model = model.lfm
        self.tokenizer = model.lfm_tokenizer
        self.pipeline = pipeline("text-generation", model=self.model, tokenizer=self.tokenizer, device=0)

    def generate_answer(self, query_and_laws):
        return self.pipeline(query_and_laws)



if __name__ == "__main__":
    query_and_ans = """法律条款：
中华人民共和国妇女权益保障法	第九章 法律责任	第七十九条 违反本法第二十二条第二款规定，未履行报告义务的，依法对直接负责的主管人员和其他直接责任人员给予处分。
第八十条 违反本法规定，对妇女实施性骚扰的，由公安机关给予批评教育或者出具告诫书，并由所在单位依法给予处分。
学校、用人单位违反本法规定，未采取必要措施预防和制止性骚扰，造成妇女权益受到侵害或者社会影响恶劣的，由上级机关或者主管部门责令改正；拒不改正或者情节严重的，依法对直接负责的主管

人格权编	第二章生命权、身体权和健康权	第一千零一十条 违背他人意愿，以言语、文字、图像、肢体行为等方式对他人实施性骚扰的，受害人有权依法请求行为人承担民事责任。
机关、企业、学校等单位应当采取合理的预防、受理投诉、调查处置等措施，防止和制止利用职权、从属关系等实施性骚扰。
第一千零一十一条 以非法拘禁等方式剥夺、限制他人的行动自由，或者非法

人格权编	第二章生命权、身体权和健康权	第一千零一十一条 以非法拘禁等方式剥夺、限制他人的行动自由，或者非法搜查他人身体的，受害人有权依法请求行为人承担民事责任。

中华人民共和国妇女权益保障法	第三章 人身和人格权益	第二十五条 用人单位应当采取下列措施预防和制止对妇女的性骚扰:
（一）制定禁止性骚扰的规章制度；
（二）明确负责机构或者人员；
（三）开展预防和制止性骚扰的教育培训活动；
（四）采
客户问题:
在某家公司中，一名员工对女同事实施了性骚扰行为，女同事向公司进行举报，但公司却没有采取必要的措施来制止这种行为。

公司未采取必要措施预防和制止性骚扰，导致女同事的权益受到侵害，该公司是否需要承担责任？"""
    t = Answer().generate_answer(query_and_ans)
    print(t)
    pass


