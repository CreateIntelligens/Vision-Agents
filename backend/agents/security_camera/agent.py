"""
Security Camera Agent - 智能監控攝影機
支援人臉辨識、物體檢測、訪客追蹤、包裹竊盜警報
"""
import asyncio
import logging
from typing import Any, Dict
from pathlib import Path
import numpy as np
import aiofiles

from vision_agents.core import Agent, User
from vision_agents.plugins import gemini, getstream
from ..base import ChatListenerProcessor
from .security_camera_processor import (
    SecurityCameraProcessor,
    PersonDetectedEvent,
    PersonDisappearedEvent,
    PackageDetectedEvent,
    PackageDisappearedEvent,
)
from .poster_generator import generate_and_post_poster

logger = logging.getLogger(__name__)

# Package theft detection delay
PACKAGE_THEFT_DELAY_SECONDS = 3.0

# Track pending theft checks
_pending_theft_tasks: Dict[str, asyncio.Task] = {}

# Track package history
_package_history: Dict[str, Dict[str, Any]] = {}

# Track last greeting time for each face (avoid spam)
_last_greeting_time: Dict[str, float] = {}
GREETING_COOLDOWN_SECONDS = 60.0  # 同一個人 60 秒內只歡迎一次


async def handle_package_theft(
    agent: Agent,
    llm,  # Gemini Realtime LLM for voice output
    face_image: np.ndarray,
    suspect_name: str,
    processor: SecurityCameraProcessor,
) -> None:
    """Generate a wanted poster and display it in the call."""
    # 使用 Gemini Realtime 直接說話
    await llm.simple_response(text=f"警報！包裹被 {suspect_name} 拿走了！正在產生通緝海報。")

    poster_bytes, tweet_url = await generate_and_post_poster(
        face_image,
        suspect_name,
        post_to_x_enabled=False,  # Disable X posting for now
        tweet_caption=f'🚨 通緝：{suspect_name} 涉嫌拿走包裹！',
    )

    if poster_bytes:
        save_path = Path(f"/tmp/wanted_poster_{suspect_name}.png")
        # 使用異步寫入避免阻塞 event loop
        async with aiofiles.open(save_path, 'wb') as f:
            await f.write(poster_bytes)
        agent.logger.info(f"✅ 通緝海報已儲存: {save_path}")

        # Share the poster in the video call for 8 seconds
        processor.share_image(poster_bytes, duration=8.0)
        await llm.simple_response(text="這是竊盜嫌疑人的通緝海報！")
    else:
        agent.logger.warning(f"⚠️ 無法產生 {suspect_name} 的通緝海報")


async def create_agent(call_id: str, user_name: str = "Human User") -> Agent:
    """創建 Security Camera Agent"""
    import os
    from dotenv import load_dotenv
    
    load_dotenv()
    logger.info(f"🎥 創建 Security Camera Agent (user={user_name})")

    # Gemini Realtime LLM with Google Search
    gemini_model = os.getenv("GEMINI_REALTIME_MODEL", "gemini-2.5-flash-native-audio-preview-12-2025")
    llm = gemini.Realtime(
        gemini_model,
        fps=2,  # 降低 FPS 減少運算負擔
        enable_google_search=True,
    )

    # Create security camera processor
    security_processor = SecurityCameraProcessor(
        fps=2,  # 降低 FPS 減少運算負擔
        time_window=1800,  # 30 minutes
        thumbnail_size=80,
        detection_interval=3.0,  # 增加偵測間隔減少負擔
        bbox_update_interval=0.5,  # 減少 bbox 更新頻率
        model_path="yolo11n.pt",  # 使用通用 YOLO 模型 (可偵測 suitcase, backpack 等)
        package_conf_threshold=0.7,
        max_tracked_packages=1,
        face_match_tolerance=0.7,  # 提高容錯度,讓同一張臉在不同角度/光線下也能識別
        person_disappeared_threshold=3.0,  # 連續 3 秒沒看到才判斷離開
    )

    # 建立 Agent
    import datetime
    current_time = datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8)))

    agent = Agent(
        edge=getstream.Edge(),
        agent_user=User(name="安保 AI", id="agent"),
        instructions=f"""你是一個居家安全監控助手，用繁體中文回答問題。你被動、簡潔，只在被詢問或需要回應時才說話。

**用戶資訊**：
- 用戶的名字是：{user_name}
- 用戶問「我的名字是什麼」或類似問題時，你要回答：{user_name}
- 用戶說話時，直接叫他 {user_name}，不要用臉部 ID

**時間資訊**：
- 現在時間：{current_time.strftime('%Y-%m-%d %H:%M:%S')}
- 時區：UTC+8（台灣/香港）
- 報告時間時要用台灣時間 UTC+8

## 你的功能

你可以使用這些工具：
- **活動紀錄**：查詢人員進出、包裹偵測等事件。用戶問「發生了什麼？」或「有人來過嗎？」時使用 `get_activity_log`
- **訪客追蹤**：用 `get_visitor_count` 和 `get_visitor_details` 查詢訪客資訊
- **包裹追蹤**：用 `get_package_count` 和 `get_package_details` 查詢包裹資訊
- **臉部記憶**：用戶說「記住我叫[名字]」或「我叫[名字]」時，使用 `remember_my_face` 記住他們。用 `get_known_faces` 查詢已認識的人

## 你的行為

- 用活動紀錄回答過去發生了什麼
- 認出熟人時，用他們的名字問候（用已註冊的名字，不要用臉部 ID）
- 用戶要求記住他們時，使用 `remember_my_face` 功能
- 回答要簡短自然，用繁體中文
- 永遠叫主要用戶（和你說話的人）的名字 {user_name}

## 重要規則

如果回應需要調用功能，你仍然要和用戶說話。獲得功能結果後，一定要用繁體中文給出回應。不要沉默地調用功能。""",
        llm=llm,
        processors=[ChatListenerProcessor("SecurityChatListener"), security_processor],
    )

    # Merge processor events with agent events
    agent.events.merge(security_processor.events)

    # Register function: get visitor count
    @llm.register_function(
        description="取得最近30分鐘偵測到的不重複訪客數量"
    )
    async def get_visitor_count() -> Dict[str, Any]:
        count = security_processor.get_visitor_count()
        state = security_processor.state()
        return {
            "unique_visitors": count,
            "total_detections": state["total_face_detections"],
            "time_window": f"{state['time_window_minutes']} 分鐘",
            "last_detection": state["last_face_detection_time"],
        }

    # Register function: get visitor details
    @llm.register_function(
        description="取得所有訪客的詳細資訊，包括首次和最後出現時間"
    )
    async def get_visitor_details() -> Dict[str, Any]:
        details = security_processor.get_visitor_details()
        return {
            "visitors": details,
            "total_unique_visitors": len(details),
        }

    # Register function: get package count
    @llm.register_function(
        description="取得包裹統計，包括目前可見和已被拿走的包裹數量"
    )
    async def get_package_count() -> Dict[str, Any]:
        currently_visible = security_processor.get_package_count()
        total_seen = len(_package_history)
        picked_up = sum(1 for p in _package_history.values() if p.get("picked_up_by"))
        return {
            "currently_visible_packages": currently_visible,
            "total_packages_seen": total_seen,
            "packages_picked_up": picked_up,
        }

    # Register function: get package details
    @llm.register_function(
        description="取得所有包裹的詳細歷史記錄，包括誰拿走了包裹"
    )
    async def get_package_details() -> Dict[str, Any]:
        return {
            "packages": list(_package_history.values()),
            "total_packages_seen": len(_package_history),
        }

    # Register function: get activity log
    @llm.register_function(
        description="取得最近的活動記錄（人員進出、包裹偵測）。用來回答「發生什麼事？」或「有人來過嗎？」"
    )
    async def get_activity_log(limit: int = 20) -> Dict[str, Any]:
        log = security_processor.get_activity_log(limit=limit)
        return {"activity_log": log, "total_entries": len(log)}

    # Register function: remember face
    @llm.register_function(
        description="記住當前人臉並給予名字，未來可以識別。當用戶說「記住我叫[名字]」或「我叫[名字]」時使用。傳入要記住的名字。"
    )
    async def remember_my_face(name: str) -> Dict[str, Any]:
        result = security_processor.register_current_face_as(name)
        return result

    # Register function: get known faces
    @llm.register_function(
        description="取得所有已註冊可識別的人臉列表"
    )
    async def get_known_faces() -> Dict[str, Any]:
        faces = security_processor.get_known_faces()
        return {"known_faces": faces, "total_known": len(faces)}

    # Subscribe to person detected event
    @agent.events.subscribe
    async def on_person_detected(event: PersonDetectedEvent):
        import time
        current_time = time.time()
        
        if event.is_new:
            agent.logger.info(f"🚨 新訪客警報: {event.face_id} 偵測到！")
            # Greet new visitors
            if hasattr(event, 'name') and event.name:
                display_name = event.name
            else:
                display_name = user_name
            await llm.simple_response(text=f"{display_name}，歡迎！")
            _last_greeting_time[event.face_id] = current_time
        else:
            # Only greet if cooldown period has passed
            last_greeting = _last_greeting_time.get(event.face_id, 0)
            if current_time - last_greeting >= GREETING_COOLDOWN_SECONDS:
                agent.logger.info(f"👤 訪客回訪: {event.face_id} (已見 {event.detection_count} 次)")
                if hasattr(event, 'name') and event.name:
                    display_name = event.name
                else:
                    display_name = user_name
                await llm.simple_response(text=f"{display_name}，歡迎回來！")
                _last_greeting_time[event.face_id] = current_time
            else:
                # Silent detection (no spam)
                agent.logger.debug(f"👤 訪客偵測: {event.face_id} (冷卻中，不打招呼)")

    # Subscribe to person disappeared event
    @agent.events.subscribe
    async def on_person_disappeared(event: PersonDisappearedEvent):
        display_name = event.name or event.face_id[:8]
        agent.logger.info(f"👤 人員離開: {display_name}")

    # Subscribe to package detected event
    @agent.events.subscribe
    async def on_package_detected(event: PackageDetectedEvent):
        # Cancel all pending theft checks when package detected
        if _pending_theft_tasks:
            cancelled_ids = list(_pending_theft_tasks.keys())
            for pkg_id in cancelled_ids:
                _pending_theft_tasks[pkg_id].cancel()
                del _pending_theft_tasks[pkg_id]
            agent.logger.info(f"📦 偵測到包裹 - 取消竊盜檢查: {', '.join(cancelled_ids)}")

        # Track package in history
        if event.package_id not in _package_history:
            _package_history[event.package_id] = {
                "package_id": event.package_id,
                "first_seen": event.timestamp.isoformat(),
                "last_seen": event.timestamp.isoformat(),
                "detection_count": 1,
                "confidence": event.confidence,
                "picked_up_by": None,
            }
        else:
            _package_history[event.package_id]["last_seen"] = event.timestamp.isoformat()
            _package_history[event.package_id]["detection_count"] += 1

        if event.is_new:
            agent.logger.info(f"📦 新包裹警報: {event.package_id} (信心度: {event.confidence:.2f})")
        else:
            agent.logger.info(f"📦 包裹回歸: {event.package_id} (第 {event.detection_count} 次)")

    # Subscribe to package disappeared event
    @agent.events.subscribe
    async def on_package_disappeared(event: PackageDisappearedEvent):
        picker_display = event.picker_name or (
            event.picker_face_id[:8] if event.picker_face_id else "未知"
        )
        agent.logger.info(
            f"📦 包裹 {event.package_id} 消失 (嫌疑人: {picker_display}) - "
            f"等待 {PACKAGE_THEFT_DELAY_SECONDS}秒確認"
        )

        async def delayed_theft_check():
            await asyncio.sleep(PACKAGE_THEFT_DELAY_SECONDS)
            # Package didn't reappear
            del _pending_theft_tasks[event.package_id]
            agent.logger.info(f"📦 包裹 {event.package_id} 確認遺失 - 觸發竊盜警報")

            # Record who picked up the package
            if event.package_id in _package_history:
                _package_history[event.package_id]["picked_up_by"] = picker_display

            if event.picker_face_id:
                face_image = security_processor.get_face_image(event.picker_face_id)
                if face_image is not None:
                    await handle_package_theft(agent, llm, face_image, picker_display, security_processor)

        _pending_theft_tasks[event.package_id] = asyncio.create_task(delayed_theft_check())

    logger.info(f"✅ Security Camera Agent 已建立 (user={user_name})")
    return agent
