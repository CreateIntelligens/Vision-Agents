#!/usr/bin/env python3
"""
Vision Agent Backend API
提供 RESTful API 來控制 Agent
"""
import os
import asyncio
import logging
import warnings
from uuid import uuid4
from urllib.parse import urlencode
from typing import Dict, Any, Optional

from dotenv import load_dotenv

# 隱藏 Stream SDK 的 dataclass warnings
warnings.filterwarnings("ignore", category=RuntimeWarning, module="dataclasses_json")
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from getstream import Stream
import threading

from vision_agents.core import User
from backend.agents import AGENT_TYPES

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 降低第三方庫的日誌等級
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("aiortc.codecs.vpx").setLevel(logging.ERROR)
logging.getLogger("libav.libvpx").setLevel(logging.CRITICAL)

app = FastAPI(title="Vision Agent API")

# CORS 設定
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 開發環境允許所有來源
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 全域狀態
active_agents: Dict[str, Dict[str, Any]] = {}
_prometheus_initialized = False
YOLO_POSE_MODEL_NAME = "yolo11n-pose.pt"


def prefetch_golf_pose_model() -> None:
    """預先下載 YOLO Pose 模型，避免啟動 Golf 範例時延遲或失敗。"""
    logger.info(f"📦 Checking YOLO pose model: {YOLO_POSE_MODEL_NAME}")
    from ultralytics import YOLO

    YOLO(YOLO_POSE_MODEL_NAME)
    logger.info(f"✅ YOLO pose model ready: {YOLO_POSE_MODEL_NAME}")


@app.on_event("startup")
async def startup_prefetch_models():
    """啟動時初始化可選功能（優雅降級）"""
    global _prometheus_initialized

    # 嘗試啟動 Prometheus（可選）
    if not _prometheus_initialized:
        try:
            from opentelemetry import metrics
            from opentelemetry.sdk.metrics import MeterProvider
            from opentelemetry.exporter.prometheus import PrometheusMetricReader

            # 不使用獨立的 HTTP server，改用 FastAPI endpoint
            reader = PrometheusMetricReader()
            provider = MeterProvider(metric_readers=[reader])
            metrics.set_meter_provider(provider)

            _prometheus_initialized = True
            logger.info("📊 Prometheus metrics enabled at /metrics")
        except ImportError:
            logger.info("ℹ️  Prometheus metrics disabled (install: pip install opentelemetry-api opentelemetry-sdk opentelemetry-exporter-prometheus prometheus-client)")
        except Exception as e:
            logger.warning(f"⚠️ Prometheus startup failed: {e}")

    # 預載 YOLO 模型（背景執行）
    threading.Thread(target=prefetch_golf_pose_model, daemon=True).start()


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
    client = Stream(api_key=api_key, api_secret=os.getenv("STREAM_API_SECRET"))

    human_id = f"user-{call_id}"
    token = client.create_token(human_id, expiration=3600)

    base_url = f"{os.getenv('EXAMPLE_BASE_URL', 'https://getstream.io/video/demos')}/join/"
    params = {
        "api_key": api_key,
        "token": token,
        "skip_lobby": "true",
        "user_name": user_name,
        "video_encoder": "h264",
        "bitrate": 12000000,
        "w": 1920,
        "h": 1080,
        "channel_type": "messaging",
    }

    return f"{base_url}{call_id}?{urlencode(params)}"


async def run_agent_in_background(call_id: str, model: str, example: str, user_name: str = "Human User"):
    """在背景執行 agent"""
    global active_agents

    # 根據 example 類型載入不同的 agent
    if example in AGENT_TYPES:
        # 使用 backend/agents 中定義的 agent
        create_agent = AGENT_TYPES[example]
        logger.info(f"🤖 Loading Agent: {example}")
        agent = await create_agent(call_id, user_name)

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
        # 其他 examples 使用 custom
        logger.warning(f"⚠️  Example '{example}' not implemented, using custom")
        agent = await AGENT_TYPES["custom"](call_id, user_name)

    # 創建 human user（在 join 之前）- 每個 call 使用唯一的 human_id
    human_id = f"user-{call_id}"
    human_user = User(name=user_name, id=human_id)  # 使用用戶輸入的名稱
    await agent.edge.create_user(user=human_user)
    logger.info(f"✅ Created human user: {human_id} with name: {user_name}")

    # 建立並加入通話
    call = await agent.create_call("default", call_id)

    # 預先創建 messaging channel 並加入 human user 作為 member
    try:
        stream_client = Stream(
            api_key=os.getenv("STREAM_API_KEY"),
            api_secret=os.getenv("STREAM_API_SECRET")
        )
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

    # 將 agent 加入 active_agents 字典
    active_agents[call_id] = {
        "agent": agent,
        "model": model,
        "call_id": call_id
    }

    try:
        async with agent.join(call):
            logger.info(f"✅ Agent joined call: {call_id}")
            await agent.finish()
    finally:
        # Agent 結束後從 active_agents 移除
        if call_id in active_agents:
            del active_agents[call_id]
            logger.info(f"🗑️  Removed agent {call_id} from active agents")


@app.get("/api/health")
async def health():
    """健康檢查"""
    return {"status": "ok"}


@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint (原始格式)"""
    try:
        from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
        from fastapi.responses import Response

        metrics_data = generate_latest()
        return Response(content=metrics_data, media_type=CONTENT_TYPE_LATEST)
    except ImportError:
        raise HTTPException(
            status_code=503,
            detail="Prometheus client not installed"
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate metrics: {str(e)}"
        )


@app.get("/api/metrics/json")
async def metrics_json():
    """返回解析後的 metrics JSON（供前端使用）"""
    try:
        from prometheus_client import generate_latest, REGISTRY

        # 收集所有 metrics
        metrics_dict = {}

        for collector in REGISTRY._collector_to_names.keys():
            for metric in collector.collect():
                metric_name = metric.name

                # 跳過內建的 process/python metrics
                if metric_name.startswith(('process_', 'python_', 'target_info')):
                    continue

                # 收集 samples
                samples = []
                for sample in metric.samples:
                    sample_dict = {
                        "name": sample.name,
                        "labels": sample.labels,
                        "value": sample.value
                    }
                    samples.append(sample_dict)

                if samples:
                    metrics_dict[metric_name] = {
                        "type": metric.type,
                        "documentation": metric.documentation,
                        "samples": samples
                    }

        return metrics_dict
    except ImportError:
        raise HTTPException(
            status_code=503,
            detail="Prometheus client not installed"
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate metrics: {str(e)}"
        )


@app.post("/api/start", response_model=StartAgentResponse)
async def start(request: StartAgentRequest):
    """啟動 Agent - 每次啟動都創建新的 Agent 實例"""
    try:
        model = request.model
        example = request.example
        user_name = request.user_name
        supported_examples = {"custom", "security_camera", "prometheus_metrics", "simple", "golf"}

        if example not in supported_examples:
            raise HTTPException(
                status_code=400,
                detail=f"Example '{example}' is not supported via /api/start",
            )

        # 產生新的 call ID
        call_id = str(uuid4())

        # 產生 Demo URL（帶入用戶名稱）
        demo_url = get_demo_url(call_id, user_name)

        # 在背景執行 agent（傳入選擇的模型、example 和用戶名稱）
        asyncio.create_task(run_agent_in_background(call_id, model, example, user_name))

        logger.info(f"🚀 Agent started with call_id: {call_id}, model: {model}, active_agents: {len(active_agents) + 1}")

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


class StopRequest(BaseModel):
    call_id: str


@app.post("/api/stop", response_model=StopResponse)
async def stop(request: StopRequest):
    """停止特定的 Agent"""
    call_id = request.call_id

    if call_id in active_agents:
        # 注意：實際上 agent.finish() 會自動清理，這裡只是標記
        logger.info(f"🛑 Stopping agent {call_id}")
        # Agent 會在 finish() 時自動從 active_agents 移除
        return StopResponse(success=True)
    else:
        logger.warning(f"⚠️  Agent {call_id} not found in active agents")
        return StopResponse(success=False)


class StatusRequest(BaseModel):
    call_id: str


@app.post("/api/status", response_model=StatusResponse)
async def status(request: StatusRequest):
    """取得特定 Agent 的狀態"""
    call_id = request.call_id

    if call_id in active_agents:
        agent_info = active_agents[call_id]
        return StatusResponse(
            running=True,
            call_id=call_id,
            model=agent_info["model"]
        )
    else:
        return StatusResponse(
            running=False,
            call_id=None,
            model=None
        )


if __name__ == '__main__':
    import uvicorn

    port = int(os.getenv('BACKEND_PORT', 8910))

    logger.info(f"\n🚀 Vision Agent Backend API 啟動中...")
    logger.info(f"📍 API Server: http://localhost:{port}\n")

    uvicorn.run(app, host="0.0.0.0", port=port)
