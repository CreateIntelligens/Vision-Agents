#!/usr/bin/env python3
"""
Vision Agent Backend API
提供 RESTful API 來控制 Agent
"""
import os
import asyncio
import logging
from uuid import uuid4
from urllib.parse import urlencode
from typing import Dict, Any, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from getstream import Stream
import threading

from vision_agents.core import Agent, User
from vision_agents.core.utils.examples import get_weather_by_location
from vision_agents.plugins import gemini, openai, getstream

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 降低 httpx 的日誌等級，避免洗版
logging.getLogger("httpx").setLevel(logging.WARNING)

# 降低 WebRTC 視訊解碼錯誤的日誌等級（網路不穩定時會有損壞的 frame）
logging.getLogger("aiortc.codecs.vpx").setLevel(logging.ERROR)
logging.getLogger("libav.libvpx").setLevel(logging.CRITICAL)

load_dotenv()

app = FastAPI(title="Vision Agent API")

# CORS 設定
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 開發環境允許所有來源
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 儲存當前運行的 agent
current_agent: Optional[Agent] = None
current_call_id: Optional[str] = None
current_llm_model: str = "gemini"
YOLO_POSE_MODEL_NAME = "yolo11n-pose.pt"


def prefetch_golf_pose_model() -> None:
    """預先下載 YOLO Pose 模型，避免啟動 Golf 範例時延遲或失敗。"""
    logger.info(f"📦 Checking YOLO pose model: {YOLO_POSE_MODEL_NAME}")
    from ultralytics import YOLO

    YOLO(YOLO_POSE_MODEL_NAME)
    logger.info(f"✅ YOLO pose model ready: {YOLO_POSE_MODEL_NAME}")


@app.on_event("startup")
async def startup_prefetch_models():
    thread = threading.Thread(target=prefetch_golf_pose_model, daemon=True)
    thread.start()


# Request/Response Models
class StartAgentRequest(BaseModel):
    model: str = "gemini"
    example: str = "custom"  # custom, simple, golf
    user_name: str = "Human User"  # 用戶名稱


class StartAgentResponse(BaseModel):
    success: bool
    call_id: str
    demo_url: str
    model: str


class StatusResponse(BaseModel):
    running: bool
    call_id: Optional[str]
    model: Optional[str]


class StopResponse(BaseModel):
    success: bool


def get_demo_url(call_id: str, user_name: str = "Human User") -> str:
    """產生 Stream Demo URL"""
    api_key = os.getenv("STREAM_API_KEY")
    api_secret = os.getenv("STREAM_API_SECRET")

    client = Stream(api_key=api_key, api_secret=api_secret)

    human_id = "user-demo-agent"
    human_name = user_name  # 使用前端傳來的名稱
    token = client.create_token(human_id, expiration=3600)

    base_url = f"{os.getenv('EXAMPLE_BASE_URL', 'https://getstream.io/video/demos')}/join/"
    params = {
        "api_key": api_key,
        "token": token,
        "skip_lobby": "true",
        "user_name": human_name,
        "video_encoder": "h264",
        "bitrate": 12000000,
        "w": 1920,
        "h": 1080,
        "channel_type": "messaging",
    }

    return f"{base_url}{call_id}?{urlencode(params)}"


async def run_agent_in_background(call_id: str, model: str, example: str):
    """在背景執行 agent"""
    global current_agent, current_call_id, current_llm_model

    # 根據 example 類型載入不同的 agent
    if example == "custom":
        # 使用我們自訂的 Agent
        from backend.agents.custom import create_agent
        logger.info(f"🤖 Using Custom Agent (Gemini Realtime)")
        agent = await create_agent(call_id)

    elif example == "simple":
        # 使用原始 example 的 create_agent
        import sys
        sys.path.insert(0, 'examples/01_simple_agent_example')
        from simple_agent_example import create_agent as create_simple_agent

        logger.info(f"🤖 Using Simple Agent Example")
        agent = await create_simple_agent()

    elif example == "golf":
        # 使用原始 Golf Coach example
        import sys
        sys.path.insert(0, 'examples/02_golf_coach_example')
        from golf_coach_example import create_agent as create_golf_agent

        logger.info(f"🤖 Using Golf Coach Example")
        agent = await create_golf_agent()

    else:
        # 其他 examples 暫時使用 custom
        logger.warning(f"⚠️  Example '{example}' not implemented yet, using custom")
        from backend.agents.custom import create_agent
        agent = await create_agent(call_id)

    # 創建 human user（在 join 之前）
    human_id = "user-demo-agent"
    human_user = User(name="Human User", id=human_id)
    await agent.edge.create_user(user=human_user)
    logger.info(f"✅ Created human user: {human_id}")

    # 建立並加入通話
    call = await agent.create_call("default", call_id)

    # 預先創建 messaging channel 並加入 human user 作為 member
    try:
        api_key = os.getenv("STREAM_API_KEY")
        api_secret = os.getenv("STREAM_API_SECRET")
        from getstream import Stream
        stream_client = Stream(api_key=api_key, api_secret=api_secret)

        # 用 server-side 權限創建 channel，加入 agent 和 human user 作為 members
        stream_client.chat.get_or_create_channel(
            type="messaging",
            id=call_id,
            data={
                "created_by_id": agent.agent_user.id,
                "members": [
                    {"user_id": agent.agent_user.id},
                    {"user_id": human_id}
                ]
            }
        )
        logger.info(f"✅ Created messaging channel with human user as member: {call_id}")
    except Exception as e:
        logger.warning(f"⚠️  Could not create messaging channel: {e}")

    current_agent = agent
    current_llm_model = model

    async with agent.join(call):
        logger.info(f"✅ Agent joined call: {call_id}")
        await agent.finish()


@app.get("/api/health")
async def health():
    """健康檢查"""
    return {"status": "ok"}


@app.post("/api/start", response_model=StartAgentResponse)
async def start(request: StartAgentRequest):
    """啟動 Agent"""
    global current_call_id

    try:
        model = request.model
        example = request.example
        user_name = request.user_name
        supported_examples = {"custom", "simple", "golf"}

        if example not in supported_examples:
            raise HTTPException(
                status_code=400,
                detail=f"Example '{example}' is not supported via /api/start",
            )

        # 產生新的 call ID
        call_id = str(uuid4())
        current_call_id = call_id

        # 產生 Demo URL（帶入用戶名稱）
        demo_url = get_demo_url(call_id, user_name)

        # 在背景執行 agent（傳入選擇的模型和 example）
        asyncio.create_task(run_agent_in_background(call_id, model, example))

        logger.info(f"🚀 Agent started with call_id: {call_id}, model: {model}")

        return StartAgentResponse(
            success=True,
            call_id=call_id,
            demo_url=demo_url,
            model=model
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to start agent: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/stop", response_model=StopResponse)
async def stop():
    """停止 Agent"""
    global current_agent, current_call_id

    current_agent = None
    current_call_id = None

    logger.info("🛑 Agent stopped")

    return StopResponse(success=True)


@app.get("/api/status", response_model=StatusResponse)
async def status():
    """取得 Agent 狀態"""
    return StatusResponse(
        running=current_agent is not None,
        call_id=current_call_id,
        model=current_llm_model if current_agent else None
    )


if __name__ == '__main__':
    import uvicorn

    port = int(os.getenv('BACKEND_PORT', 8910))

    logger.info(f"\n🚀 Vision Agent Backend API 啟動中...")
    logger.info(f"📍 API Server: http://localhost:{port}\n")

    uvicorn.run(app, host="0.0.0.0", port=port)
