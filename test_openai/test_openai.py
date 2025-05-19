# from openai import OpenAI
# client = OpenAI(
#     base_url='https://api.claudeplus.top/v1',
#     # sk-xxx替换为自己的key
#     api_key='sk-wwbQhXG6BwoScLbXpVzf5kmSBPgt7o59l0mg2Wybqk2BuQXJ'
# )
# completion = client.chat.completions.create(
#   model="gpt-4o",
#   messages=[
#     {"role": "system", "content": "You are a helpful assistant."},
#     {"role": "user", "content": "Hello!"}
#   ]
# )
# print(completion.choices[0].message)

from openai import OpenAI
import base64
import os

client = OpenAI(
    base_url='https://api.claudeplus.top/v1',
    api_key=os.environ['OPENAI_API_KEY']
)


# 读取本地图片并进行Base64编码
def encode_image_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

# 使用本地图片路径
script_dir = os.path.dirname(os.path.abspath(__file__))
local_image_path = os.path.join(script_dir, "query_img.png")
img_base64 = encode_image_to_base64(local_image_path)


response = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "这张图片里有什么?如果我想把红色笔放到黑色笔筒里,我需要做什么?我能操作的对象是图中的数字标记点."},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{img_base64}"
                    }
                },
            ],
        }
    ],
    max_tokens=300,
)
print(response.choices[0])