import base64
import os

from dotenv import load_dotenv, find_dotenv
from openai import OpenAI

load_dotenv(find_dotenv())

client = OpenAI(
    # 若没有配置环境变量，请用阿里云百炼API Key将下行替换为：api_key="sk-xxx",
    # 各地域的API Key不同。获取API Key：https://help.aliyun.com/zh/model-studio/get-api-key
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    # 以下为华北2（北京）地域的URL，调用时请将 {WorkspaceId} 替换为真实的业务空间ID，各地域的URL不同。
    base_url="https://llm-k66cel0uwa622q6q.cn-beijing.maas.aliyuncs.com/compatible-mode/v1",
)
# f.read() -> b'\x89PNG...'
# b64encode() --> b'iVBORw0K...'
# decode() --> 'iVBORw0K...'
with open('G:\slkh\Pictures\labor\小喇叭.png', 'rb') as f:
    base64_image = base64.b64encode(f.read()).decode("utf-8")

print(base64_image)

completion = client.chat.completions.create(
    model="qwen3.7-plus",  # 此处以qwen3.7-plus为例，可按需更换模型名称。模型列表：https://help.aliyun.com/zh/model-studio/models
    messages=[
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                         "url": f"data:image/jpeg;base64,{base64_image}"
                    },
                },
                {"type": "text", "text": "图中描绘的是什么景象?"},
            ],
        },
    ],
)
print(completion.choices[0].message.content)
