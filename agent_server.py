import os
import json
from typing import List, Dict, Optional, Any
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Depends, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
# 导入你的模型工厂
from model_factory import (
    create_model_by_key,
    create_emb_model_by_key,
    model_map,
    ChatModel
)

# 加载环境变量
load_dotenv()

# 初始化 FastAPI 应用
app = FastAPI(
    title="AI Agent Server",
    description="基于自定义模型工厂的多模型 AI Agent 服务器",
    version="1.0.0"
)

# 配置跨域
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 数据模型定义
class ChatRequest(BaseModel):
    """对话请求模型"""
    message: str
    model_key: str = "ds_v3"  # 使用 model_factory 的 key
    session_id: str = "default"
    system_prompt: Optional[str] = None
    temperature: float = 0.5
    think: bool = False  # 是否启用思考模式
    max_tokens: int = 8192

class ChatResponse(BaseModel):
    """对话响应模型"""
    response: str
    model_key: str
    session_id: str
    usage: Optional[Dict[str, float]] = None

class EmbeddingRequest(BaseModel):
    """嵌入请求模型"""
    texts: List[str]
    model_key: str = "Qwen_emb"

class EmbeddingResponse(BaseModel):
    """嵌入响应模型"""
    embeddings: List[List[float]]
    model_key: str

# 会话记忆存储（内存版）
memory_store: Dict[str, List[Dict[str, str]]] = {}

# WebSocket 连接管理器
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def send_personal_message(self, message: str, websocket: WebSocket):
        await websocket.send_text(message)

manager = ConnectionManager()

# 获取会话历史
def get_session_history(session_id: str) -> List[Dict[str, str]]:
    if session_id not in memory_store:
        memory_store[session_id] = []
    return memory_store[session_id]

# 核心对话接口
@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """使用自定义模型工厂的对话接口"""
    try:
        # 1. 验证模型 key 是否存在
        if request.model_key not in model_map:
            raise HTTPException(status_code=400, detail=f"不支持的模型 key: {request.model_key}")
        
        # 2. 创建模型实例
        chat_model = create_model_by_key(
            key=request.model_key,
            think=request.think,
            tools=None  # 可根据需求传入工具配置
        )
        
        # 3. 构建消息（包含会话历史）
        session_history = get_session_history(request.session_id)
        messages = []
        
        # 添加系统提示
        if request.system_prompt:
            messages.append({"role": "system", "content": request.system_prompt})
        else:
            messages.append({"role": "system", "content": "You are a helpful ai assistant."})
        
        # 添加历史对话
        messages.extend(session_history)
        
        # 添加当前用户消息
        messages.append({"role": "user", "content": request.message})
        
        # 4. 调用模型获取响应
        response_msg = chat_model.get_response_by_msg(
            messages=messages,
            temperature=request.temperature,
            max_tokens=request.max_tokens
        )
        response_content = response_msg.content
        
        # 5. 更新会话历史
        session_history.append({"role": "user", "content": request.message})
        session_history.append({"role": "assistant", "content": response_content})
        
        # 6. 返回响应
        return ChatResponse(
            response=response_content,
            model_key=request.model_key,
            session_id=request.session_id
        )
    
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"服务器内部错误: {str(e)}")

# 嵌入接口
@app.post("/api/embeddings", response_model=EmbeddingResponse)
async def get_embeddings(request: EmbeddingRequest):
    """生成文本嵌入向量"""
    try:
        if request.model_key not in model_map:
            raise HTTPException(status_code=400, detail=f"不支持的嵌入模型 key: {request.model_key}")
        
        # 创建嵌入模型实例
        emb_model = create_emb_model_by_key(request.model_key)
        
        # 生成嵌入向量
        embeddings = emb_model.get_embeddings(request.texts)
        
        return EmbeddingResponse(
            embeddings=embeddings,
            model_key=request.model_key
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"生成嵌入失败: {str(e)}")

# 清除会话记忆
@app.delete("/api/clear-memory/{session_id}")
async def clear_memory(session_id: str):
    if session_id in memory_store:
        del memory_store[session_id]
        return {"message": f"会话 {session_id} 记忆已清除"}
    raise HTTPException(status_code=404, detail="会话不存在")

# 获取支持的模型列表
@app.get("/api/models")
async def list_supported_models():
    """获取所有配置的模型列表"""
    return {
        "chat_models": list(model_map.keys()),
        "embedding_models": ["Qwen_emb"],  # 可根据实际情况调整
        "model_details": model_map
    }

# WebSocket 对话接口
@app.websocket("/ws/chat/{session_id}")
async def websocket_chat(websocket: WebSocket, session_id: str):
    await manager.connect(websocket)
    try:
        while True:
            # 接收客户端消息
            data = await websocket.receive_json()
            model_key = data.get("model_key", "ds_v3")
            message = data.get("message")
            think = data.get("think", False)
            
            # 创建模型实例
            chat_model = create_model_by_key(model_key, think=think)
            
            # 构建消息
            session_history = get_session_history(session_id)
            messages = [{"role": "system", "content": "You are a helpful ai assistant."}]
            messages.extend(session_history)
            messages.append({"role": "user", "content": message})
            
            # 获取响应
            response_msg = chat_model.get_response_by_msg(messages=messages)
            response_content = response_msg.content
            
            # 更新历史
            session_history.append({"role": "user", "content": message})
            session_history.append({"role": "assistant", "content": response_content})
            
            # 发送响应
            await manager.send_personal_message(
                json.dumps({
                    "response": response_content,
                    "model_key": model_key,
                    "session_id": session_id
                }),
                websocket
            )
    except WebSocketDisconnect:
        manager.disconnect(websocket)

# 健康检查
@app.get("/health")
async def health_check():
    return {"status": "healthy", "supported_models": list(model_map.keys())}

# 启动服务
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "agent_server:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )