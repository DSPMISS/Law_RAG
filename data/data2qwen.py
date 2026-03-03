import json
from typing import List, Dict, Any


def transform_data_item(item: Dict[str, Any]) -> Dict[str, List[Dict[str, str]]]:
    """
    转换单个数据项为所需的messages格式

    Args:
        item: 包含reference, question, answer的字典

    Returns:
        包含messages列表的字典
    """
    # 提取数据
    references = item.get("reference", [])
    question = item.get("question", "")
    answer = item.get("answer", "")
    if answer == "":
        return -1

    # 处理references，去除多余的引号和逗号
    cleaned_references = []
    for ref in references:
        # 清理字符串：去除两端的空格、换行、多余的引号和逗号
        ref = ref.strip()
        if ref.endswith(','):
            ref = ref[:-1]
        if ref.endswith('\n'):
            ref = ref[:-1]
        cleaned_references.append(ref)

    # 构建法律条款字符串
    law_clauses = "\n".join(cleaned_references)

    # 构建完整的消息列表
    messages = [
        {
            "role": "system",
            "content": "你是一个法律助手，请用人类的语气根据给出的相关法律条款回答客户问题"
        },
        {
            "role": "user",
            "content": f"""法律条款：
{law_clauses}

客户问题：
{question}"""
        },
        {
            "role": "assistant",
            "content": answer
        }
    ]

    return {"messages": messages}


def transform_data_list(data_list: List[Dict[str, Any]]) -> List[Dict[str, List[Dict[str, str]]]]:
    """
    转换整个数据列表

    Args:
        data_list: 原始数据列表

    Returns:
        转换后的数据列表
    """
    transformed_list = []
    for item in data_list:
        transformed_item = transform_data_item(item)
        if transformed_item == -1:
            continue
        transformed_list.append(transformed_item)
    return transformed_list


def process_file(input_file: str, output_file: str):
    """
    处理整个文件：读取、转换、保存

    Args:
        input_file: 输入文件路径
        output_file: 输出文件路径
    """
    # 读取原始数据（根据实际文件格式调整）
    with open(input_file, 'r', encoding='utf-8') as f:
        if input_file.endswith('.jsonl'):
            data_list = json.load(f)
        else:
            # 如果是文本文件，按行读取并解析
            content = f.read()
            # 这里假设文件内容是有效的Python列表
            data_list = eval(content)

    # 转换数据
    transformed_data = transform_data_list(data_list)

    # 保存转换后的数据
    i = 0
    with open(output_file, 'a+', encoding='utf-8') as f:
        for item in transformed_data:
            # 每行一个JSON对象
            json_line = json.dumps(item, ensure_ascii=False)
            f.write(json_line + '\n')


    print(f"转换完成！共处理 {len(transformed_data)} 条数据")
    print(f"已保存到: {output_file}")

    # 显示第一条数据作为示例
    if transformed_data:
        print("\n第一条数据示例:")
        print(json.dumps(transformed_data[0], ensure_ascii=False, indent=2))

process_file("真实场景法律咨询/训练数据_带法律依据_92k.json", "真实场景法律咨询/data2qwen.jsonl")
