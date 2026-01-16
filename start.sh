#!/bin/bash

# Vision Agents 快速啟動腳本

set -e

echo "🚀 Vision Agents - Docker 快速啟動"
echo "=================================="
echo ""

# 檢查 .env 檔案
if [ ! -f ".env" ]; then
    echo "❌ 錯誤：找不到 .env 檔案"
    echo ""
    echo "請執行以下步驟："
    echo "1. cp .env.example .env"
    echo "2. 編輯 .env 填入你的 API keys"
    echo ""
    exit 1
fi

# 檢查必要的環境變數
source .env

if [ -z "$STREAM_API_KEY" ] || [ "$STREAM_API_KEY" = "your_stream_api_key_here" ]; then
    echo "❌ 錯誤：STREAM_API_KEY 未設定"
    echo "請編輯 .env 填入你的 Stream API Key"
    echo ""
    exit 1
fi

if [ -z "$STREAM_API_SECRET" ] || [ "$STREAM_API_SECRET" = "your_stream_api_secret_here" ]; then
    echo "❌ 錯誤：STREAM_API_SECRET 未設定"
    echo "請編輯 .env 填入你的 Stream API Secret"
    echo ""
    exit 1
fi

if [ -z "$GOOGLE_API_KEY" ] || [ "$GOOGLE_API_KEY" = "your_google_api_key_here" ]; then
    echo "⚠️  警告：GOOGLE_API_KEY 未設定"
    echo "Agent 可能無法正常運作"
    echo ""
fi

# 檢查 Docker
if ! command -v docker &> /dev/null; then
    echo "❌ 錯誤：Docker 未安裝"
    echo "請先安裝 Docker：https://docs.docker.com/get-docker/"
    exit 1
fi

if ! docker compose version &> /dev/null; then
    echo "❌ 錯誤：Docker Compose 未安裝"
    echo "請先安裝 Docker Compose"
    exit 1
fi

# 停止舊容器
echo "🛑 停止舊容器（如果存在）..."
docker compose down 2>/dev/null || true
echo ""

# 建置 image（如果需要）
if ! docker images | grep -q "vision-agents.*latest"; then
    echo "🔨 首次執行，建置 Docker image..."
    echo "（這可能需要幾分鐘）"
    docker compose build backend
    echo ""
fi

# 啟動服務
echo "🚀 啟動服務（Nginx + Backend + Frontend）..."
docker compose up -d nginx backend frontend

# 等待服務啟動
echo ""
echo "⏳ 等待服務啟動..."
sleep 5

# 檢查容器狀態
if docker ps | grep -q "vision-agent-nginx"; then
    echo ""
    echo "✅ Vision Agent 已成功啟動！"
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "📍 請開啟瀏覽器訪問："
    echo ""
    echo "   🔒 https://localhost:8910"
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""
    echo "⚠️  首次訪問會看到安全警告（自簽證書）"
    echo "    請點擊「進階」→「繼續前往」"
    echo ""
    echo "💡 使用提示："
    echo "   • 查看 Nginx 日誌：docker logs -f vision-agent-nginx"
    echo "   • 查看 Backend 日誌：docker logs -f vision-agent-backend"
    echo "   • 查看 Frontend 日誌：docker logs -f vision-agent-frontend"
    echo "   • 停止服務：docker compose down"
    echo "   • 重啟服務：docker compose restart"
    echo ""
else
    echo ""
    echo "❌ 啟動失敗！"
    echo ""
    echo "請執行以下指令查看日誌："
    echo "docker logs vision-agent-nginx"
    echo "docker logs vision-agent-backend"
    echo "docker logs vision-agent-frontend"
    echo ""
    exit 1
fi
