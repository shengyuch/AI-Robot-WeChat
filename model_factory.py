from openai import OpenAI
# from mem0 import Memory
from typing import List

# 通用注册表/工厂，用于根据 key 生成对象
class Registry:
    def __init__(self):
        self._creators = {}

    def register(self, key, creator):
        self._creators[key] = creator

    def create(self, key, *args, **kwargs):
        if key not in self._creators:
            raise KeyError(f"未知的 key: {key}")
        return self._creators[key](*args, **kwargs)

model_map = {
    "qwen2.5_vl": {
        "base_url": "http://172.30.209.139:15503/v1",
        "base_api_key": "EMPTY",
        "model_name": "Qwen2.5-VL-72B",
    },
    "qwen2.5_72": {
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1/",
        "base_api_key": "sk-1f95c2cdd80d4ff2955e28d734f3c3ff",
        "model_name": "qwen2.5-72b-instruct"
    },
    "qwen_next": {
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1/",
        "base_api_key": "sk-cb34bc779d114fcca051c8bb8e1e1f9a",
        "model_name": "qwen3-next-80b-a3b-instruct"
    },
    "qwen3_32": {
        "base_url": "http://117.50.89.227:4217/v1",
        "base_api_key": "EMPTY",
        "model_name": "Qwen",
    },
    "sparkX1": {
        "base_url": "http://26.0.128.230:9971/v1",
        "base_api_key": "EMPTY",
        "model_name": "",
    },
    "ds_v3": {
        "base_url": "http://maas-api.cn-huabei-1.xf-yun.com/v1/",
        "base_api_key": "sk-BmzvnaYJsYzxbdme31A98491DaF84b8f957fB33f47C203Df",
        "model_name": "xdeepseekv3",
    },
    "Qwen_emb": {
        "base_url": "http://117.50.89.227:7001/v1",
        "base_api_key": "EMPTY",
        "model_name": "Qwen_emb",
    },
    "Qwen3_235B": {
            "base_url": "http://117.50.89.227:4216/v1",
            "base_api_key": "sk-Gxvxu4mbQMhkMwDL93B64dA6958b421bA39b2dAfBfFd3a32",
            "model_name": "Qwen/Qwen3-235B-A22B-Thinking-2507",
    },
    "qwen_plus": {
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "base_api_key": "sk-1f95c2cdd80d4ff2955e28d734f3c3ff",
        "model_name": "qwen-plus",
    }
}

class EmbModel:
    def __init__(
        self, base_url: str, api_key: str, model_name: str, think: bool = False
    ):
        self._client = OpenAI(base_url=base_url, api_key=api_key)
        self._model_name = model_name

    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        if isinstance(texts, str):
            texts = [texts]
        try:
            response = self._client.embeddings.create(
                model=self._model_name, input=texts
            )
            return [item.embedding for item in response.data]
        except Exception as e:
            raise RuntimeError(f"Failed to fetch embeddings: {e}")

class ChatModel:
    def __init__(
        self, base_url: str, api_key: str, model_name: str, think: bool = False, tools: List[dict] = None
    ):
        self._client = OpenAI(base_url=base_url, api_key=api_key)
        self._model_name = model_name
        self._think = think
        self.tools = tools

    def get_response(
        self,
        prompt: str,
        system_prompt: str = "You are a helpful ai assistant.",
        temperature: float = 0.5,
        max_tokens: int = 8192
    ):
        response = self._client.chat.completions.create(
            model=self._model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            extra_body={"chat_template_kwargs": {"enable_thinking": self._think}},
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content

    def get_response_by_msg(
        self,
        messages: list,
        tool_choices: str = "auto",
        temperature: float = 0.5,
        max_tokens: int = 8192,
    ):
        response = self._client.chat.completions.create(
            model=self._model_name,
            messages=messages,
            tools=self.tools if self.tools else None,
            tool_choice=tool_choices if self.tools else None,
            extra_body={"chat_template_kwargs": {"enable_thinking": self._think}},
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return response.choices[0].message

# 构建模型工厂：根据 key 返回 ChatModel 实例
model_factory = Registry()
emb_model_factory = Registry()

def _chat_model_creator_factory(key: str, think: bool = False, tools: list=None):
    cfg = model_map[key]
    return ChatModel(
        base_url=cfg["base_url"],
        api_key=cfg["base_api_key"],
        model_name=cfg["model_name"],
        think=think,
        tools=tools
    )

def _emb_model_creator_factory(key: str):
    cfg = model_map[key]
    return EmbModel(
        base_url=cfg["base_url"],
        api_key=cfg["base_api_key"],
        model_name=cfg["model_name"],
    )

# 预注册所有配置中的 key
for _key in model_map.keys():
    # 使用闭包保存 key，并支持传递 think 参数
    def make_chat_creator(key):
        def creator(*args, **kwargs):
            think = kwargs.pop("think", False) if "think" in kwargs else False
            tools = kwargs.pop("tools", None) if "tools" in kwargs else None

            return _chat_model_creator_factory(key, think=think, tools=tools)

        return creator

    def make_emb_creator(key):
        def creator(*args, **kwargs):
            return _emb_model_creator_factory(key)

        return creator

    model_factory.register(_key, make_chat_creator(_key))
    emb_model_factory.register(_key, make_emb_creator(_key))

def create_model_by_key(key: str, think: bool = False, tools: list=None) -> ChatModel:
    """根据 key 创建一个已配置好的聊天模型对象。

    用法：
        model = create_model_by_key("ds_v3")
        print(model.get_response("你好"))
    """
    return model_factory.create(key, think=think, tools=tools)

# # 为了兼容旧用法，保留一个默认模型与函数
# _default_key = "ds_v3"
# _default_model = create_model_by_key(_default_key, think=False)

# def get_response(prompt):
#     return _default_model.get_response(prompt)

def create_emb_model_by_key(key: str) -> EmbModel:
    """根据 key 创建一个已配置好的 embedding 模型对象。

    用法：
        emb_model = create_emb_model_by_key("Qwen_emb")
        embedding = emb_model.get_embeddings("测试文本")
    """
    return emb_model_factory.create(key)

if __name__ == "__main__":
    # 1. 创建聊天模型并对话
    chat_model = create_model_by_key("qwen2.5_72", think=False)
    response = chat_model.get_response("介绍一下通义千问2.5")
    print(response)

    # 2. 创建嵌入模型并生成向量
    emb_model = create_emb_model_by_key("Qwen_emb")
    embeddings = emb_model.get_embeddings(["测试文本1", "测试文本2"])
    print(embeddings)

    # 3. 多轮消息对话（支持工具调用）
    messages = [
        {"role": "system", "content": "你是一个计算器"},
        {"role": "user", "content": "1+1等于多少"}
    ]
    msg_response = chat_model.get_response_by_msg(messages)
    print(msg_response.content)