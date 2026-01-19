"""
自訂 Agent - 繁體中文語音助理
使用 Gemini 2.5 Flash Realtime，支援視訊、RAG 知識庫和天氣查詢
"""
import asyncio
import logging
from pathlib import Path
from typing import Dict, Any
from vision_agents.core import Agent, User
from vision_agents.core.utils.examples import get_weather_by_location
from vision_agents.plugins import gemini, getstream
from vision_agents.core.processors import Processor
import datetime

logger = logging.getLogger(__name__)

# 知識庫路徑
KNOWLEDGE_DIR = Path(__file__).parent.parent / "knowledge"


class ChatListenerProcessor(Processor):
    """
    監聽 Stream Chat 訊息並轉發給 Gemini Realtime 的處理器。
    直接監聽 Stream Channel 的訊息事件。
    """
    def __init__(self):
        self.agent = None
        self._task = None
        self._processed_message_ids = set()
        self._channel = None

    def attach_agent(self, agent):
        self.agent = agent

    async def start(self):
        logger.info("🎧 ChatListenerProcessor started - 監聽用戶文字輸入")
        self._task = asyncio.create_task(self._listen_loop())

    async def stop(self):
        logger.info("🛑 ChatListenerProcessor stopped")
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

    async def close(self):
        await self.stop()

    @property
    def name(self) -> str:
        return "ChatListener"

    async def _listen_loop(self):
        """監聽新的文字訊息並發送給 Gemini"""
        # 等待 conversation 和 channel 初始化
        while not self.agent.conversation:
            if getattr(self.agent, 'closed', False):
                return
            await asyncio.sleep(0.5)

        # 取得 Stream Channel
        if hasattr(self.agent.conversation, 'channel'):
            self._channel = self.agent.conversation.channel
            logger.info("🎧 ChatListener 已連接到 Stream Channel，開始監聽用戶文字輸入")
        else:
            logger.error("❌ Conversation 沒有 channel 屬性")
            return

        # 使用輪詢方式檢查新訊息
        # 直接使用 channel.client.query_channels() 來獲取訊息
        logger.info("✅ 開始輪詢 Stream Chat 訊息（每 0.5 秒檢查一次）")

        # Channel ID 就是 call_id（在創建 channel 時設定的）
        # 從 agent 的 call 物件取得
        try:
            if hasattr(self.agent, 'call') and self.agent.call:
                channel_id = self.agent.call.id
                channel_type = "messaging"
                logger.info(f"📍 監聽 channel: type={channel_type}, id={channel_id}")
            else:
                logger.error("❌ Agent 沒有 call 物件")
                return
        except Exception as e:
            logger.error(f"❌ 無法取得 channel 資訊: {e}")
            import traceback
            logger.error(f"詳細錯誤: {traceback.format_exc()}")
            return

        while True:
            try:
                await asyncio.sleep(0.5)

                # 使用 client.query_channels 查詢這個 channel 的訊息
                try:
                    response = await self._channel.client.query_channels(
                        filter_conditions={
                            "type": channel_type,
                            "id": channel_id
                        },
                        message_limit=10
                    )

                    if response.data.channels and len(response.data.channels) > 0:
                        channel_data = response.data.channels[0]
                        messages = channel_data.messages if hasattr(channel_data, 'messages') else []

                        # 只在有新訊息時才印 log
                        new_messages = [msg for msg in messages if msg.id not in self._processed_message_ids]
                        if new_messages:
                            logger.info(f"🔍 查詢到 {len(messages)} 條訊息，{len(new_messages)} 條是新的")

                        for msg in messages:
                            message_id = msg.id
                            user_id = msg.user.id if msg.user else None
                            text = msg.text or ""

                            # 跳過已處理的訊息
                            if message_id in self._processed_message_ids:
                                continue

                            self._processed_message_ids.add(message_id)

                            # 跳過 agent 自己的訊息
                            if user_id == self.agent.agent_user.id:
                                logger.debug(f"⏭️  跳過 agent 訊息 (user_id={user_id})")
                                continue

                            # 跳過空訊息
                            if not text or not text.strip():
                                logger.debug(f"⏭️  跳過空訊息")
                                continue

                            # 跳過語音轉文字產生的訊息（這些會有 custom.chunk_group 標記）
                            if hasattr(msg, 'custom') and msg.custom and 'chunk_group' in msg.custom:
                                logger.debug(f"⏭️  跳過語音轉文字訊息（chunk_group={msg.custom.get('chunk_group')}）")
                                continue

                            logger.info(f"📩 收到用戶文字訊息: {text}")

                            # 等待 1 秒讓最新的視訊幀先被發送（fps=2，所以至少會有 2 幀更新）
                            await asyncio.sleep(1.0)

                            # 發送給 Gemini Realtime
                            try:
                                await self.agent.llm.simple_response(text=text)
                                logger.info(f"✅ 已將文字發送給 Gemini: {text}")
                            except Exception as e:
                                logger.error(f"❌ 發送文字給 Gemini 失敗: {e}")

                except Exception as e:
                    logger.error(f"查詢訊息時出錯: {e}")
                    import traceback
                    logger.error(f"詳細錯誤: {traceback.format_exc()}")
                    await asyncio.sleep(1.0)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Chat listener loop error: {e}")
                await asyncio.sleep(1.0)


async def sync_knowledge_store(file_search_store: gemini.GeminiFilesearchRAG) -> None:
    """
    同步本地和雲端知識庫文件。
    - 檢查本地和雲端文件是否一致
    - 如果不一致，重建 store（最簡單的方式實現完全同步）
    """
    # 取得本地文件清單
    local_files = {f.name: f for f in KNOWLEDGE_DIR.glob("*") if f.is_file()}
    local_filenames = set(local_files.keys())

    # 取得雲端文件清單
    remote_filenames = set(file_search_store._uploaded_files)

    logger.info(f"📊 本地文件: {local_filenames}")
    logger.info(f"📊 雲端文件: {remote_filenames}")

    # 檢查是否需要同步
    files_to_delete = remote_filenames - local_filenames
    files_to_upload = local_filenames - remote_filenames

    if not files_to_delete and not files_to_upload:
        logger.info("✅ 本地和雲端文件已同步")
        return

    # 如果有文件需要刪除或新增，重建 store
    if files_to_delete or files_to_upload:
        logger.info(f"🔄 偵測到文件變更 (刪除: {files_to_delete}, 新增: {files_to_upload})")
        logger.info(f"🗑️  刪除舊 store: {file_search_store._store_name}")

        await file_search_store.clear()

        logger.info(f"✨ 建立新 store...")
        await file_search_store.create()

        logger.info(f"📤 上傳所有本地文件...")
        await file_search_store.add_directory(KNOWLEDGE_DIR)

        logger.info(f"✅ 完成同步 (新增: {files_to_upload}, 刪除: {files_to_delete})")


async def create_agent(call_id: str, user_name: str = "Human User") -> Agent:
    """建立自訂 Agent，包含 RAG 知識庫"""

    # 初始化 Gemini File Search（RAG）
    logger.info("📚 初始化 Gemini File Search...")
    file_search_store = await gemini.create_file_search_store(
        name="custom_agent_rag",
        knowledge_dir=KNOWLEDGE_DIR,
    )

    # 同步本地和雲端文件
    logger.info("🔄 同步本地和雲端文件...")
    await sync_knowledge_store(file_search_store)

    # 使用 Gemini Realtime（支援視訊）
    llm = gemini.Realtime(
        "gemini-2.5-flash-native-audio-preview-12-2025",
        fps=2,  # 提高到 2 FPS 減少延遲（JPEG 壓縮後記憶體使用約 360-720MB/小時）
    )

    agent = Agent(
        edge=getstream.Edge(),
        agent_user=User(name="AI 助理", id="agent"),
        instructions=f"""你是一個友善的繁體中文語音 AI 助理，具有視訊分析能力。

**用戶資訊**：
- 用戶的名字是：{user_name}
- 當用戶問你「我的名字是什麼」或類似問題時，你應該回答：{user_name}

**視訊分析能力（最重要）**：
- 你可以即時看到用戶的視訊畫面
- 當用戶問「你看到什麼？」、「這是什麼？」、「畫面上有什麼？」時，你必須分析當下的視訊畫面並回答
- 你看到的是即時畫面，每秒更新 2 次
- 永遠基於「當下最新的畫面」來回答，不要參考過去的畫面

你可以：
1. **視訊分析** - 分析用戶的即時視訊畫面，描述看到的物體、場景、文字等
2. 回答關於 Vision Agents 框架的問題 - 使用 search_knowledge 函數搜索知識庫
3. 查詢任何位置的天氣 - 使用 get_weather 函數

重要規則：
- **當用戶問關於畫面的問題時，立即分析最新的視訊幀，不要說「我看不到」或參考舊畫面**
- 當用戶詢問 Vision Agents 相關問題時，必須先呼叫 search_knowledge 搜索知識庫
- 請用繁體中文回答，保持簡短、對話式的風格
- 不要使用特殊符號或格式，保持親切友善

範例：
- 用戶問「你看到什麼？」→ 分析當下視訊畫面並描述
- 用戶問「Vision Agents 支援哪些模型？」→ 呼叫 search_knowledge("Vision Agents 支援的模型")
- 用戶問「台北天氣如何？」→ 呼叫 get_weather("台北")
- 用戶問「我的名字是什麼？」→ 回答：{user_name}""",
        llm=llm,
        processors=[ChatListenerProcessor()],
    )

    # 註冊知識庫搜索功能
    @llm.register_function(description="搜索 Vision Agents 框架的知識庫，查詢關於框架功能、支援的模型、應用場景、使用方法等相關資訊。當用戶詢問任何關於 Vision Agents 的問題時必須使用此函數。")
    async def search_knowledge(query: str) -> str:
        try:
            results = await file_search_store.search(query, top_k=3)
            return results if results else "知識庫中找不到相關資訊。"
        except Exception as e:
            logger.error(f"知識庫搜索出錯: {e}")
            return f"搜索出錯: {str(e)}"

    # 註冊天氣查詢功能
    @llm.register_function(description="取得指定位置的天氣資訊")
    async def get_weather(location: str) -> Dict[str, Any]:
        return await get_weather_by_location(location)

    logger.info(f"✅ 自訂 Agent 已建立（RAG + 天氣查詢啟用）")

    return agent
