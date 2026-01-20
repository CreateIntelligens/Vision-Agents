"""
Prometheus Metrics Agent - 性能監控
支援 OpenTelemetry + Prometheus 即時指標收集
"""
import logging
from vision_agents.core import Agent, User
from vision_agents.plugins import gemini, getstream

logger = logging.getLogger(__name__)


async def create_agent(call_id: str, user_name: str = "Human User") -> Agent:
    """創建 Prometheus Metrics Agent"""
    logger.info(f"📊 創建 Prometheus Metrics Agent (user={user_name})")

    # Gemini Realtime 內建 STT/TTS，不需要額外設定
    llm = gemini.Realtime(
        "gemini-2.5-flash-native-audio-preview-12-2025",
        fps=0,  # 不需要視訊
    )

    # 建立 Agent（Gemini Realtime 自帶語音能力）
    agent = Agent(
        edge=getstream.Edge(),
        agent_user=User(name="監控 AI", id="agent"),
        llm=llm,
        instructions=f"""你是一個 AI 性能監控助理，專注於系統指標和性能分析。

**用戶資訊**：
- 用戶的名字是：{user_name}

**你的功能**：
1. **即時監控** - 你的所有互動都在被監控（LLM 延遲、Token 用量等）
2. **性能分析** - 可以討論系統性能、延遲等話題
3. **指標說明** - 解釋各種監控指標的含義

**監控的指標包括**：
- LLM 響應延遲（latency_ms）
- 首個 token 時間（time_to_first_token_ms）
- 輸入/輸出 tokens 數量
- 工具調用次數和延遲
- 語音識別/合成的延遲和時長

**重要規則**：
- 保持簡短、專業的回答
- 用繁體中文回答
- 對性能話題保持專業態度

範例：
- 用戶問「現在性能如何？」→ 說明當前正在收集指標
- 用戶問「什麼是 token？」→ 解釋 LLM tokens 的概念
- 用戶問「你好」→ 友善回應並簡介監控功能""",
    )

    # 嘗試啟用 MetricsCollector（可選）
    try:
        from vision_agents.core.observability import MetricsCollector
        _ = MetricsCollector(agent)
        logger.info("📈 MetricsCollector 已啟用")
    except Exception as e:
        logger.warning(f"⚠️ MetricsCollector 啟用失敗（可選功能）: {e}")

    logger.info(f"✅ Prometheus Metrics Agent 已建立 (user={user_name})")
    return agent
