"""
Security Camera Agent - 智能監控攝影機
支援人臉辨識、物體檢測、訪客追蹤
"""
import logging
from vision_agents.core import Agent, User
from vision_agents.plugins import gemini, getstream
from ..base import ChatListenerProcessor

logger = logging.getLogger(__name__)


async def create_agent(call_id: str, user_name: str = "Human User") -> Agent:
    """創建 Security Camera Agent"""
    logger.info(f"🎥 創建 Security Camera Agent (user={user_name})")

    # Gemini Realtime LLM - 5 FPS 用於更頻繁的畫面分析
    llm = gemini.Realtime(
        "gemini-2.5-flash-native-audio-preview-12-2025",
        fps=5,
        enable_google_search=True,
    )

    # 建立 Agent
    agent = Agent(
        edge=getstream.Edge(),
        agent_user=User(name="安保 AI", id="agent"),
        instructions=f"""你是一個專業的安保 AI 助理，具有即時視訊監控能力。

**用戶資訊**：
- 用戶的名字是：{user_name}

**視訊分析能力**：
- 你每秒接收 5 次視訊畫面更新
- 你可以分析畫面中的人物、物體、活動
- 永遠基於「當下最新的畫面」來回答

**功能**：
1. **訪客監控** - 追蹤進出的人員，記錄訪客
2. **物體檢測** - 識別包裹、車輛等物體
3. **異常偵測** - 發現可疑活動並提醒
4. **即時查詢** - 使用 Google Search 查詢即時資訊

**重要規則**：
- 當用戶問關於畫面的問題時，立即分析最新的視訊幀
- 保持警覺，主動報告重要事件
- 用繁體中文回答，保持專業但友善的語氣

範例：
- 用戶問「現在有人嗎？」→ 分析畫面並回答
- 用戶問「看到什麼可疑的嗎？」→ 仔細檢查畫面並報告
- 檢測到新訪客 → 主動說「有新訪客到達」""",
        llm=llm,
        processors=[ChatListenerProcessor("SecurityChatListener")],
    )

    logger.info(f"✅ Security Camera Agent 已建立 (user={user_name})")
    return agent
