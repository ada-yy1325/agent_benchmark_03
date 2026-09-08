import os
from dotenv import load_dotenv
from openai import OpenAI

# 1. 强制加载当前项目目录下的 .env 文件
load_dotenv()

# 2. 安全地读取刚才配置的环境变量
client = OpenAI(
    base_url=os.getenv("DEEPSEEK_BASE_URL"),
    api_key=os.getenv("DEEPSEEK_API_KEY")
)

# 3. 这里的 model 名字需要改成你服务商实际支持的名称（比如 deepseek-chat）
# 你的原代码里写的是 claude-sonnet，请根据你的中转代理商要求修改
response = client.chat.completions.create(
    model="deepseek-v4-flash", 
    messages=[
        {"role": "user", "content": "repeat me:hello"}
    ]
)

print(response.choices[0].message.content)