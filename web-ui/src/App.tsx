import { useState, useEffect } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { Sparkles, Video, Loader2, Power, PowerOff } from 'lucide-react'
import './App.css'

type AgentStatus = {
  running: boolean
  call_id: string | null
  model: string | null
}

type ModelType = 'gemini' | 'openai'
type ExampleType = 'custom' | 'simple' | 'golf'

const EXAMPLES = [
  { value: 'custom', label: '自訂 Agent（繁體中文語音助理）', description: 'Gemini 2.5 Flash Realtime - 支援視訊與天氣查詢' },
  { value: 'simple', label: 'Simple Agent（原始範例）', description: 'Deepgram + ElevenLabs + Gemini - 英文語音助理' },
  { value: 'golf', label: 'Golf Coach（高爾夫教練）', description: 'Gemini Realtime 視訊教練 - 分析高爾夫揮桿動作' },
]

function App() {
  const [status, setStatus] = useState<AgentStatus>({
    running: false,
    call_id: null,
    model: null,
  })
  const [selectedExample, setSelectedExample] = useState<ExampleType>('custom')
  const [loading, setLoading] = useState(false)
  const [demoUrl, setDemoUrl] = useState<string | null>(null)

  // 定期檢查狀態
  useEffect(() => {
    const checkStatus = async () => {
      try {
        const res = await fetch('/api/status')
        const data = await res.json()
        setStatus(data)
      } catch (error) {
        console.error('Failed to check status:', error)
      }
    }

    checkStatus()
    const interval = setInterval(checkStatus, 5000)
    return () => clearInterval(interval)
  }, [])

  const startAgent = async () => {
    setLoading(true)
    try {
      const res = await fetch('/api/start', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          model: 'gemini',  // 統一使用 Gemini（支援視訊）
          example: selectedExample
        }),
      })
      const data = await res.json()

      if (data.success) {
        setDemoUrl(data.demo_url)
        setStatus({
          running: true,
          call_id: data.call_id,
          model: data.model,
        })
      } else {
        alert('啟動失敗: ' + data.error)
      }
    } catch (error) {
      alert('啟動失敗: ' + error)
    } finally {
      setLoading(false)
    }
  }

  const stopAgent = async () => {
    try {
      await fetch('/api/stop', { method: 'POST' })
      setStatus({ running: false, call_id: null, model: null })
      setDemoUrl(null)
    } catch (error) {
      console.error('Failed to stop agent:', error)
    }
  }

  const openDemo = () => {
    if (demoUrl) {
      window.open(demoUrl, '_blank')
    }
  }

  return (
    <motion.div
      className="container"
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5 }}
    >
      <motion.div
        className="header"
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ delay: 0.2 }}
      >
        <Sparkles className="icon" size={32} />
        <h1>Vision Agent</h1>
      </motion.div>

      <motion.div
        className={`status ${status.running ? 'running' : 'idle'}`}
        initial={{ scale: 0.9 }}
        animate={{ scale: 1 }}
        transition={{ delay: 0.3 }}
      >
        {status.running ? (
          <>
            <div className="status-dot" />
            Agent 運行中
          </>
        ) : (
          'Agent 待機中'
        )}
      </motion.div>

      <AnimatePresence mode="wait">
        {!status.running ? (
          <motion.div
            key="start-section"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
          >
            <div className="model-select-box">
              <h3>📦 選擇 Agent 範例</h3>
              <select
                className="model-select example-select"
                value={selectedExample}
                onChange={(e) => setSelectedExample(e.target.value as ExampleType)}
                disabled={loading}
              >
                {EXAMPLES.map((example) => (
                  <option key={example.value} value={example.value}>
                    {example.label}
                  </option>
                ))}
              </select>
              <p className="example-description">
                {EXAMPLES.find(e => e.value === selectedExample)?.description}
              </p>
            </div>

            <motion.button
              className="btn btn-primary"
              onClick={startAgent}
              disabled={loading}
              whileHover={{ scale: 1.02 }}
              whileTap={{ scale: 0.98 }}
            >
              {loading ? (
                <>
                  <Loader2 className="spin" size={20} />
                  啟動中...
                </>
              ) : (
                <>
                  <Power size={20} />
                  啟動 Agent
                </>
              )}
            </motion.button>
          </motion.div>
        ) : (
          <motion.div
            key="running-section"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
          >
            <div className="info-box">
              <h3>📞 連線資訊</h3>
              <p>Call ID:</p>
              <div className="call-id">{status.call_id}</div>
              <p className="model-info">
                模型: {status.model === 'gemini' ? 'Gemini 2.5 Flash' : 'OpenAI GPT-4o'}
              </p>
              <motion.button
                className="btn btn-primary"
                onClick={openDemo}
                whileHover={{ scale: 1.02 }}
                whileTap={{ scale: 0.98 }}
              >
                <Video size={20} />
                開啟視訊通話
              </motion.button>
            </div>

            <motion.button
              className="btn btn-danger"
              onClick={stopAgent}
              whileHover={{ scale: 1.02 }}
              whileTap={{ scale: 0.98 }}
            >
              <PowerOff size={20} />
              停止 Agent
            </motion.button>
          </motion.div>
        )}
      </AnimatePresence>

      <div className="info-box usage">
        <h3>💡 使用說明</h3>
        <ol>
          <li>選擇 AI 模型（Gemini 支援視訊，OpenAI 僅語音）</li>
          <li>點擊「啟動 Agent」開始</li>
          <li>等待 Agent 準備完成</li>
          <li>點擊「開啟視訊通話」進入通話介面</li>
          <li>在瀏覽器中允許麥克風和攝影機權限</li>
          <li>開始與 AI Agent 對話！</li>
        </ol>
      </div>
    </motion.div>
  )
}

export default App
