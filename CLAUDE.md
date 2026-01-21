# Claude AI 開發指引

這份文件記錄了 Vision Agents 專案的開發規範和最佳實踐，供 Claude AI 參考。

## Agent 開發規範

### 1. Gemini Realtime 配置（標準）

所有 Vision Agent 應使用統一的 Gemini Realtime 配置：

```python
import os
from dotenv import load_dotenv

load_dotenv()
gemini_model = os.getenv("GEMINI_REALTIME_MODEL", "gemini-2.5-flash-native-audio-preview-12-2025")

llm = gemini.Realtime(
    gemini_model,
    fps=2,  # 標準 FPS：2 (平衡性能與資源)
    enable_google_search=True,  # 啟用 Google Search
)
```

**關鍵要點**：
- ✅ **必須使用環境變數** `GEMINI_REALTIME_MODEL` 來指定模型
- ✅ **FPS 統一設為 2**（除非有特殊需求）
- ✅ **啟用 Google Search**（enable_google_search=True）
- ✅ **從 .env 載入配置**（load_dotenv()）

### 2. Instructions 語言規範

所有 Agent 的 instructions **必須使用繁體中文**撰寫：

```python
agent = Agent(
    edge=getstream.Edge(),
    agent_user=User(name="AI 助理", id="agent"),
    instructions=f"""你是一個友善的 AI 助理，用繁體中文回答問題。

**用戶資訊**：
- 用戶的名字是：{user_name}
- 當用戶問「我的名字是什麼」時，你要回答：{user_name}

**你的功能**：
1. 功能描述（繁體中文）
2. 行為規則（繁體中文）

**重要規則**：
- 保持簡短、自然的回答
- 用繁體中文思考和回應
""",
    llm=llm,
)
```

**為什麼使用繁體中文 instructions？**
- ❌ 英文 instructions → AI 用英文邏輯思考 → 翻譯成中文 → 回答怪異
- ✅ 繁體中文 instructions → AI 用中文邏輯思考 → 直接回答 → 自然流暢

### 3. 用戶名稱處理

每個 Agent 都必須正確處理用戶名稱：

```python
async def create_agent(call_id: str, user_name: str = "Human User") -> Agent:
    # 在 instructions 中嵌入 user_name
    instructions = f"""你的用戶名字是：{user_name}"""

    # 在事件處理中使用 user_name
    @agent.events.subscribe
    async def on_event(event):
        await llm.simple_response(text=f"你好，{user_name}！")
```

**backend/app.py 中的用戶設置**：
```python
# 創建用戶（Vision Agents 層）
human_id = f"user-{call_id}"
human_user = User(name=user_name, id=human_id)
await agent.edge.create_user(user=human_user)

# 更新 Stream Chat 顯示名稱
stream_client.upsert_users(
    UserRequest(id=human_id, name=user_name),
    UserRequest(id=agent.agent_user.id, name=agent.agent_user.name)
)
```

### 4. Agent 命名規範

```python
agent_user=User(name="<角色名稱>", id="agent")
```

**範例**：
- Security Camera: `name="安保 AI"`
- Custom Agent: `name="AI 助理"`
- Prometheus Metrics: `name="監控 AI"`

### 5. 環境變數配置

所有模型配置應放在 `.env` 中：

```bash
# Gemini model for Vision Agents
GEMINI_REALTIME_MODEL=gemini-2.5-flash-native-audio-preview-12-2025
```

## Docker 配置規範

### Dockerfile 最佳實踐

**依賴應在 Dockerfile 中安裝**，而非 docker-compose.yml 的啟動命令：

```dockerfile
# ✅ 正確：在 Dockerfile 安裝
RUN apt-get update && apt-get install -y cmake build-essential
COPY pyproject.toml uv.lock ./
COPY agents-core ./agents-core
COPY plugins ./plugins
RUN uv sync --frozen
RUN uv pip install fastapi uvicorn[standard] ...

# ❌ 錯誤：在 docker-compose.yml 每次啟動都安裝
command: >
  bash -c "
    apt-get update && apt-get install -y cmake &&
    uv pip install fastapi ...
  "
```

**優點**：
- 更快的啟動速度
- Docker 層緩存
- 符合最佳實踐

### docker-compose.yml 簡化

Backend 啟動命令應該簡潔：

```yaml
command: >
  bash -c "
    cd backend && python app.py
  "
```

## 常見問題

### Q: 為什麼 FPS 統一設為 2？
**A**: 平衡性能與資源消耗。FPS=5 太高會導致：
- 事件循環阻塞
- 音訊延遲警告
- 記憶體使用增加

FPS=2 已足夠視訊分析需求。

### Q: 何時可以使用 fps=0？
**A**: 只有在**完全不需要視訊**的 Agent 才使用 fps=0。但為了統一規範，建議都用 fps=2。

### Q: enable_google_search 一定要開嗎？
**A**: 是的，這讓 Agent 能查詢即時資訊（天氣、新聞等），提升實用性。

## 版本歷史

- 2025-01-21: 初始版本，定義 Agent 開發規範
