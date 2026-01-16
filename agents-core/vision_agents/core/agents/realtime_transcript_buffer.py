"""
Realtime Transcript Buffer

緩衝 Realtime LLM 的語音轉文字輸出，將短片段累積成完整句子後再發送。
解決 Gemini Realtime 每次只輸出一個字導致聊天室訊息分散的問題。
"""

import asyncio
import logging
from typing import Callable, Awaitable, Optional
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class RealtimeTranscriptBuffer:
    """
    緩衝 Realtime LLM 的語音轉文字片段。

    收集短時間內的文字片段，在以下情況發送：
    1. 偵測到句子結束標點（。！？.!?）
    2. 超過最大等待時間（flush_interval_ms）
    3. 手動呼叫 flush()

    Example:
        buffer = RealtimeTranscriptBuffer(
            flush_callback=send_to_chat,
            flush_interval_ms=800,
        )

        # 收到片段時呼叫
        await buffer.append("你")
        await buffer.append("好")
        await buffer.append("嗎")
        await buffer.append("？")  # 偵測到句號，自動發送 "你好嗎？"
    """

    # 發送完整訊息的 callback
    flush_callback: Callable[[str], Awaitable[None]]

    # 最大等待時間（毫秒），超過後自動發送
    flush_interval_ms: int = 800

    # 句子結束標點
    sentence_endings: str = "。！？.!?\n"

    # 內部狀態
    _buffer: str = field(default="", init=False)
    _flush_task: Optional[asyncio.Task] = field(default=None, init=False)
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock, init=False)

    async def append(self, text: str) -> None:
        """
        添加文字片段到緩衝區。

        如果偵測到句子結束標點，會立即發送緩衝的內容。
        否則會啟動計時器，超時後自動發送。
        """
        if not text:
            return

        async with self._lock:
            self._buffer += text

            # 取消之前的 flush 計時器
            if self._flush_task and not self._flush_task.done():
                self._flush_task.cancel()
                try:
                    await self._flush_task
                except asyncio.CancelledError:
                    pass

            # 檢查是否有句子結束標點
            if any(end in text for end in self.sentence_endings):
                await self._flush_internal()
            else:
                # 啟動新的 flush 計時器
                self._flush_task = asyncio.create_task(self._delayed_flush())

    async def _delayed_flush(self) -> None:
        """延遲 flush - 等待指定時間後自動發送"""
        try:
            await asyncio.sleep(self.flush_interval_ms / 1000.0)
            async with self._lock:
                await self._flush_internal()
        except asyncio.CancelledError:
            pass

    async def _flush_internal(self) -> None:
        """內部 flush 方法（不加鎖）"""
        if self._buffer:
            content = self._buffer.strip()
            self._buffer = ""
            if content:
                try:
                    await self.flush_callback(content)
                except Exception as e:
                    logger.error(f"Failed to flush transcript buffer: {e}")

    async def flush(self) -> None:
        """手動發送緩衝區內容"""
        async with self._lock:
            # 取消計時器
            if self._flush_task and not self._flush_task.done():
                self._flush_task.cancel()
                try:
                    await self._flush_task
                except asyncio.CancelledError:
                    pass
            await self._flush_internal()

    async def close(self) -> None:
        """關閉緩衝器，發送剩餘內容"""
        await self.flush()

    @property
    def pending_text(self) -> str:
        """取得目前緩衝區中的文字（不發送）"""
        return self._buffer
