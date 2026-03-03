import os

from langchain_openai import ChatOpenAI
from transformers import AutoModelForCausalLM, AutoTokenizer

api_key_qwen = os.getenv("DASHSCOPE_API_KEY")

qwen = ChatOpenAI(
    api_key=api_key_qwen,
    model="qwen2.5-7b-instruct-1m",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    temperature=0.7,
    max_tokens=100,
)

lora_qwen = ChatOpenAI(
    api_key=api_key_qwen,
    model="qwen2.5-7b-instruct-1m",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    temperature=0.7,
    max_tokens=100,
)

lfm = AutoModelForCausalLM.from_pretrained("LiquidAI/LFM2-350M",
                                           cache_dir="./LFM2",
                                           low_cpu_mem_usage=True)
lfm_tokenizer = AutoTokenizer.from_pretrained("LiquidAI/LFM2-350M",
                                              cache_dir="./LFM2",)
