"""
共用的 Processor 和工具函數
"""
import asyncio
import logging
from typing import Optional
from vision_agents.core.processors import Processor

logger = logging.getLogger(__name__)


class ChatListenerProcessor(Processor):
    """
    監聽 Stream Chat 訊息並轉發給 LLM 的處理器。
    可被不同的 Agent 共用。
    """
    def __init__(self, processor_name: str = "ChatListener"):
        self._processor_name = processor_name
        self.agent = None
        self._task: Optional[asyncio.Task] = None
        self._processed_message_ids: set = set()
        self._channel = None

    @property
    def name(self) -> str:
        return self._processor_name

    def attach_agent(self, agent):
        self.agent = agent

    async def start(self):
        logger.info(f"🎧 {self.name} started - 監聽用戶文字輸入")
        self._task = asyncio.create_task(self._listen_loop())

    async def stop(self):
        logger.info(f"🛑 {self.name} stopped")
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

    async def close(self):
        await self.stop()

    async def _listen_loop(self):
        """監聽新的文字訊息並發送給 LLM"""
        # 等待 conversation 和 channel 初始化
        while not self.agent.conversation:
            if getattr(self.agent, 'closed', False):
                return
            await asyncio.sleep(0.5)

        # 取得 Stream Channel
        if hasattr(self.agent.conversation, 'channel'):
            self._channel = self.agent.conversation.channel
            logger.info(f"🎧 {self.name} 已連接到 Stream Channel")
        else:
            logger.error("❌ Conversation 沒有 channel 屬性")
            return

        # Channel ID
        try:
            if hasattr(self.agent, 'call') and self.agent.call:
                channel_id = self.agent.call.id
                channel_type = "messaging"
            else:
                logger.error("❌ Agent 沒有 call 物件")
                return
        except Exception as e:
            logger.error(f"❌ 無法取得 channel 資訊: {e}")
            return

        while True:
            try:
                await asyncio.sleep(0.5)

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

                        new_messages = [msg for msg in messages if msg.id not in self._processed_message_ids]
                        if new_messages:
                            logger.info(f"🔍 查詢到 {len(messages)} 條訊息，{len(new_messages)} 條是新的")

                        for msg in messages:
                            if msg.id in self._processed_message_ids:
                                continue

                            self._processed_message_ids.add(msg.id)
                            user_id = msg.user.id if msg.user else None
                            text = msg.text or ""

                            # 跳過 agent 自己的訊息
                            if user_id == self.agent.agent_user.id:
                                continue

                            # 跳過空訊息
                            if not text or not text.strip():
                                continue

                            # 跳過語音轉文字產生的訊息
                            if hasattr(msg, 'custom') and msg.custom and 'chunk_group' in msg.custom:
                                continue

                            logger.info(f"📩 收到用戶文字訊息: {text}")

                            # 等待讓最新的視訊幀先被發送
                            await asyncio.sleep(1.0)

                            # 發送給 LLM
                            try:
                                await self.agent.llm.simple_response(text=text)
                                logger.info(f"✅ 已將文字發送給 LLM: {text}")
                            except Exception as e:
                                logger.error(f"❌ 發送文字給 LLM 失敗: {e}")

                except Exception as e:
                    logger.debug(f"查詢訊息時出錯: {e}")
                    await asyncio.sleep(1.0)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Chat listener loop error: {e}")
                await asyncio.sleep(1.0)
