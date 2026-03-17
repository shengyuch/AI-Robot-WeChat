import os
import json
import requests
from typing import Optional
from dotenv import load_dotenv
from wechaty import Wechaty, Contact, Message, Room
from wechaty_puppet import FileBox

# 加载环境变量
load_dotenv()

# 配置项
FASTAPI_URL = os.getenv("FASTAPI_URL", "http://127.0.0.1:8000")
DEFAULT_MODEL_KEY = os.getenv("DEFAULT_MODEL_KEY", "ds_v3")
# 机器人开关（可通过指令控制）
BOT_ENABLED = True

# 初始化微信机器人
bot = Wechaty()

def call_ai_api(
    message: str,
    user_id: str,
    model_key: str = DEFAULT_MODEL_KEY,
    system_prompt: Optional[str] = None
) -> str:
    """调用 FastAPI AI 接口"""
    try:
        # 构建请求数据
        data = {
            "message": message,
            "model_key": model_key,
            "session_id": f"wechat_{user_id}",  # 会话ID绑定微信用户ID
            "system_prompt": system_prompt,
            "temperature": 0.5,
            "think": False
        }
        
        # 调用接口
        response = requests.post(
            url=f"{FASTAPI_URL}/api/chat",
            json=data,
            timeout=60  # 超时时间
        )
        
        # 处理响应
        if response.status_code == 200:
            result = response.json()
            return result.get("response", "抱歉，我没有理解你的意思。")
        else:
            return f"接口调用失败：{response.status_code} - {response.text}"
    
    except requests.exceptions.ConnectionError:
        return "无法连接到 AI 服务器，请检查服务器是否启动。"
    except requests.exceptions.Timeout:
        return "请求超时，请稍后再试。"
    except Exception as e:
        return f"发生错误：{str(e)}"

@bot.on_message
async def on_message(msg: Message):
    """消息处理逻辑"""
    global BOT_ENABLED
    
    # 忽略自己发送的消息、群聊消息（如需支持群聊可注释）
    if msg.self() or msg.room():
        return
    
    # 获取发送者信息
    talker: Contact = msg.talker()
    user_id = talker.id
    user_name = talker.name
    message_text = msg.text().strip()
    
    # 指令处理
    if message_text.startswith("#"):
        if message_text == "#关闭":
            BOT_ENABLED = False
            await msg.say("🤖 机器人已关闭，不再回复消息。")
            return
        elif message_text == "#开启":
            BOT_ENABLED = True
            await msg.say("🤖 机器人已开启，我可以正常回复消息啦！")
            return
        elif message_text.startswith("#模型"):
            # 切换模型，示例：#模型 qwen2.5_72
            parts = message_text.split()
            if len(parts) >= 2:
                new_model = parts[1]
                # 验证模型是否存在
                try:
                    model_list = requests.get(f"{FASTAPI_URL}/api/models").json()
                    if new_model in model_list.get("chat_models", []):
                        global DEFAULT_MODEL_KEY
                        DEFAULT_MODEL_KEY = new_model
                        await msg.say(f"🤖 模型已切换为：{new_model}")
                    else:
                        await msg.say(f"❌ 不支持的模型：{new_model}\n支持的模型：{model_list.get('chat_models')}")
                except:
                    await msg.say("❌ 无法获取模型列表，请检查服务器连接。")
            else:
                await msg.say("使用方法：#模型 模型名称\n示例：#模型 qwen2.5_72")
            return
        elif message_text == "#清除记忆":
            # 清除会话记忆
            try:
                requests.delete(f"{FASTAPI_URL}/api/clear-memory/wechat_{user_id}")
                await msg.say("🧹 已清除你的对话记忆。")
            except:
                await msg.say("❌ 清除记忆失败。")
            return
        elif message_text == "#帮助":
            help_text = """
🤖 微信 AI 机器人使用帮助：
#开启 - 开启机器人
#关闭 - 关闭机器人
#模型 [模型名] - 切换AI模型（示例：#模型 qwen2.5_72）
#清除记忆 - 清除你的对话历史
#帮助 - 查看帮助信息
            """.strip()
            await msg.say(help_text)
            return
    
    # 如果机器人已关闭，不处理消息
    if not BOT_ENABLED:
        return
    
    # 忽略空消息
    if not message_text:
        return
    
    # 处理文本消息
    if msg.type() == bot.Message.Type.MESSAGE_TYPE_TEXT:
        # 正在思考提示
        await msg.say("🤔 正在思考中...")
        
        # 调用AI接口
        ai_response = call_ai_api(
            message=message_text,
            user_id=user_id,
            model_key=DEFAULT_MODEL_KEY
        )
        
        # 发送回复
        await msg.say(ai_response)
    
    # 处理非文本消息（图片、语音等）
    else:
        await msg.say("🤖 目前仅支持文本消息哦～")

@bot.on_login
async def on_login(user: Contact):
    """登录成功回调"""
    print(f"✅ 微信机器人登录成功：{user.name}")
    print(f"🔗 AI 服务器地址：{FASTAPI_URL}")
    print(f"📌 默认模型：{DEFAULT_MODEL_KEY}")

@bot.on_logout
async def on_logout(user: Contact):
    """登出回调"""
    print(f"❌ 微信机器人已登出：{user.name}")

if __name__ == "__main__":
    # 启动机器人
    print("🚀 启动微信机器人...")
    print("📱 请扫描二维码登录微信")
    bot.start()