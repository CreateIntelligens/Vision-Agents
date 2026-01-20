#!/usr/bin/env python3
"""
測試 Gemini Live API 並保存音訊到檔案
"""
import asyncio
import os
import wave
from google import genai
from google.genai.types import LiveConnectConfig, PrebuiltVoiceConfig

async def test_and_save_audio():
    """測試 Gemini 並保存音訊"""

    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("❌ 錯誤：請設定 GOOGLE_API_KEY 環境變數")
        return

    print("🧪 開始測試 Gemini Live API 並保存音訊...")
    print(f"📍 API Key: {api_key[:10]}...")

    client = genai.Client(api_key=api_key)
    model = "gemini-2.5-flash-native-audio-preview-12-2025"

    print(f"🤖 使用模型: {model}")

    config = LiveConnectConfig(
        response_modalities=["AUDIO"],
        speech_config=PrebuiltVoiceConfig(voice_name="Aoede"),
    )

    print("🔗 連接到 Gemini Live API...")

    # 用來存儲所有音訊資料
    audio_chunks = []

    try:
        async with client.aio.live.connect(model=model, config=config) as session:
            print("✅ 已連接到 Gemini")

            # 發送簡單的中文請求
            print("📤 發送測試訊息: '你好，請用中文說一段話'")
            await session.send(input="你好，請用中文說一段話", end_of_turn=True)

            print("👂 等待 Gemini 回應並收集音訊...")

            async for response in session.receive():
                # 收集音訊資料
                if response.data:
                    audio_chunks.append(response.data)
                    print(f"🔊 收到音訊片段: {len(response.data)} bytes")

                # 檢查 server_content 中的 inline_data
                if hasattr(response, 'server_content') and response.server_content:
                    if hasattr(response.server_content, 'model_turn'):
                        model_turn = response.server_content.model_turn
                        if hasattr(model_turn, 'parts'):
                            for part in model_turn.parts:
                                if hasattr(part, 'inline_data') and part.inline_data:
                                    if hasattr(part.inline_data, 'data') and part.inline_data.data:
                                        audio_chunks.append(part.inline_data.data)
                                        print(f"🔊 收到音訊片段 (inline_data): {len(part.inline_data.data)} bytes")
                                if hasattr(part, 'text') and part.text:
                                    print(f"📝 文字: {part.text[:100]}...")

                # 收到完整回應後結束
                if response.server_content and hasattr(response.server_content, 'turn_complete'):
                    if response.server_content.turn_complete:
                        print("✅ Gemini 回應完成")
                        break

            # 保存音訊到檔案
            if audio_chunks:
                total_size = sum(len(chunk) for chunk in audio_chunks)
                print(f"\n📊 收到 {len(audio_chunks)} 個音訊片段，總大小: {total_size} bytes")

                # 合併所有音訊片段
                audio_data = b''.join(audio_chunks)

                # Gemini 返回的是 PCM 格式，24kHz，16-bit，單聲道
                # 保存為 WAV 檔案
                output_file = "/tmp/gemini_audio_test.wav"
                with wave.open(output_file, 'wb') as wav_file:
                    wav_file.setnchannels(1)  # 單聲道
                    wav_file.setsampwidth(2)  # 16-bit = 2 bytes
                    wav_file.setframerate(24000)  # 24kHz
                    wav_file.writeframes(audio_data)

                print(f"\n✅ 音訊已保存到: {output_file}")
                print(f"📁 檔案大小: {len(audio_data)} bytes")
                print(f"⏱️  音訊長度: {len(audio_data) / (24000 * 2):.2f} 秒")
                print(f"\n🎧 請用播放器測試: aplay {output_file}")
            else:
                print("\n❌ 沒有收到任何音訊資料")

    except Exception as e:
        print(f"❌ 測試失敗: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_and_save_audio())
