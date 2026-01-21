"""
Prometheus Metrics Agent - 性能監控
支援 OpenTelemetry + Prometheus 即時指標收集
"""
import logging
from vision_agents.core import Agent, User
from vision_agents.plugins import gemini, getstream
from ..base import ChatListenerProcessor

logger = logging.getLogger(__name__)


async def create_agent(call_id: str, user_name: str = "Human User") -> Agent:
    """創建 Prometheus Metrics Agent"""
    import os
    from dotenv import load_dotenv

    load_dotenv()
    logger.info(f"📊 創建 Prometheus Metrics Agent (user={user_name})")

    # 環境變數
    gemini_model = os.getenv("GEMINI_REALTIME_MODEL", "gemini-2.5-flash-native-audio-preview-12-2025")
    public_host = os.getenv("PUBLIC_HOST", "localhost")
    web_port = os.getenv("WEB_UI_PORT", "8910")
    llm = gemini.Realtime(
        gemini_model,
        fps=2,  # 標準 FPS 設定
        enable_google_search=True,  # 啟用 Google Search
    )

    # 建立 Agent（Gemini Realtime 自帶語音能力）
    agent = Agent(
        edge=getstream.Edge(),
        agent_user=User(name="監控 AI", id="agent"),
        llm=llm,
        processors=[ChatListenerProcessor("PrometheusMetricsChatListener")],
        instructions=f"""你是一個專業的系統監控 AI，負責協助用戶了解 Vision Agents 的性能指標。

**用戶資訊**：
- 用戶的名字是：{user_name}

**你的功能**：
1. 當用戶詢問「系統性能如何」、「有什麼指標」、「監控數據」時，調用 `get_current_metrics()` 查詢實際數據
2. 解釋指標的含義
3. 提供性能優化建議

**🚨 關鍵規則 - 絕對不能違反 🚨**：
- **永遠只報告 `get_current_metrics()` 返回的真實數據**
- **如果函數返回 "no_data_yet"，就說「系統剛啟動，還沒有數據」**
- **如果某個指標不存在，說「這個指標目前沒有記錄」**
- **絕對不能編造、猜測、或虛構任何數值**
- 用繁體中文回答，保持專業

**當前系統實際收集的指標**：
- `realtime_transcriptions_agent`: AI 語音輸出次數
- `realtime_transcriptions_user`: 用戶語音輸入次數
- `realtime_audio_bytes`: 音訊輸出位元組數
- `llm_tool_calls`: LLM 工具調用次數
- `llm_tool_latency_ms`: LLM 工具調用延遲
- `getstream_requests`: Stream API 請求次數
- `process_resident_memory_bytes`: 進程記憶體使用量
- `process_cpu_seconds`: CPU 使用時間

**視覺化儀表板**：
- 獨立監控頁面：https://{public_host}:{web_port}/metrics
- JSON API：https://{public_host}:{web_port}/api/metrics/json
- Prometheus 格式：https://{public_host}:{web_port}/api/metrics/prometheus

記住：**只報告真實存在的數據，不編造！**
""",
    )

    # 嘗試啟用 MetricsCollector（可選）
    try:
        from vision_agents.core.observability import MetricsCollector
        _ = MetricsCollector(agent)
        logger.info("📈 MetricsCollector 已啟用")
        logger.info("=" * 60)
        logger.info("📊 Prometheus Metrics Agent")
        logger.info("=" * 60)
        logger.info("監控端點：")
        logger.info(f"  - 儀表板: https://{public_host}:{web_port}/metrics")
        logger.info(f"  - JSON API: https://{public_host}:{web_port}/api/metrics/json")
        logger.info(f"  - Prometheus: https://{public_host}:{web_port}/api/metrics/prometheus")
        logger.info("")
        logger.info("收集的指標：")
        logger.info("  - realtime_transcriptions_agent/user (語音轉錄)")
        logger.info("  - realtime_audio_bytes (音訊輸出)")
        logger.info("  - llm_tool_calls, llm_tool_latency_ms (工具調用)")
        logger.info("  - getstream_requests (API 請求)")
        logger.info("  - process_resident_memory_bytes, process_cpu_seconds (系統)")
        logger.info("=" * 60)
    except Exception as e:
        logger.warning(f"⚠️ MetricsCollector 啟用失敗（可選功能）: {e}")

    # 註冊查詢指標的 function
    @llm.register_function(
        description="查詢當前的性能監控指標，包括：realtime_transcriptions_agent（AI語音輸出次數）、realtime_transcriptions_user（用戶語音輸入次數）、realtime_audio_bytes（音訊位元組）、llm_tool_calls（工具調用次數）、getstream_requests（API請求）、process_resident_memory_bytes（記憶體）、process_cpu_seconds（CPU時間）"
    )
    async def get_current_metrics() -> dict:
        """從 Prometheus Registry 讀取當前指標"""
        try:
            from prometheus_client import REGISTRY

            result = {
                "status": "collecting",
                "metrics": {}
            }

            # 我們關心的指標前綴
            METRIC_PREFIXES = ('llm_', 'stt_', 'tts_', 'turn_', 'realtime_', 'getstream_', 'process_')

            # 讀取所有 collector 的指標
            for collector in list(REGISTRY._collector_to_names.keys()):
                for metric in collector.collect():
                    # 過濾我們關心的指標
                    if not metric.name.startswith(METRIC_PREFIXES):
                        continue

                    for sample in metric.samples:
                        metric_name = sample.name

                        # 如果是 histogram/summary 的 sum/count，計算平均值
                        if sample.name.endswith('_sum'):
                            count_name = sample.name.replace('_sum', '_count')
                            count_sample = next((s for s in metric.samples if s.name == count_name), None)
                            if count_sample and count_sample.value > 0:
                                avg_value = sample.value / count_sample.value
                                base_name = sample.name.replace('_sum', '')
                                result["metrics"][base_name + "_avg"] = round(avg_value, 2)
                        # 顯示原始值（跳過 bucket 和 count）
                        elif not sample.name.endswith('_bucket') and not sample.name.endswith('_count'):
                            # 轉換記憶體為 MB 方便閱讀
                            if 'memory_bytes' in metric_name:
                                result["metrics"][metric_name.replace('_bytes', '_mb')] = round(sample.value / 1024 / 1024, 2)
                            # 轉換音訊為 MB
                            elif metric_name == 'realtime_audio_bytes':
                                result["metrics"]["realtime_audio_mb"] = round(sample.value / 1024 / 1024, 2)
                            else:
                                result["metrics"][metric_name] = round(sample.value, 2)

            if not result["metrics"]:
                result["status"] = "no_data_yet"
                result["message"] = "系統剛啟動，尚未收集到數據。請與我對話幾次後再查詢。"
            else:
                result["status"] = "success"
                result["total_metrics"] = len(result["metrics"])

            return result
        except Exception as e:
            logger.error(f"查詢指標失敗: {e}")
            return {
                "status": "error",
                "message": f"查詢失敗: {str(e)}"
            }

    logger.info(f"✅ Prometheus Metrics Agent 已建立 (user={user_name})")
    return agent
