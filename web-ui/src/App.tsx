import { useState, useEffect } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { Sparkles, Video, Loader2, Power, PowerOff, Bot, User, Phone, Lightbulb, AlertCircle, CheckCircle } from 'lucide-react'
import './App.css'

type AgentStatus = {
  running: boolean
  call_id: string | null
  model: string | null
}

function App() {
  const [status, setStatus] = useState<AgentStatus>({
    running: false,
    call_id: null,
    model: null,
  })
  const [loading, setLoading] = useState(false)
  const [demoUrl, setDemoUrl] = useState<string | null>(null)
  const [userName, setUserName] = useState('')
  const [nameError, setNameError] = useState('')

  // 定期檢查狀態
  useEffect(() => {
    const checkStatus = async () => {
      // 如果沒有 call_id，跳過檢查
      if (!status.call_id) return

      try {
        const res = await fetch('/api/status', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ call_id: status.call_id }),
        })
        const data = await res.json()
        setStatus(data)
      } catch (error) {
        console.error('Failed to check status:', error)
      }
    }

    if (status.call_id) {
      checkStatus()
      const interval = setInterval(checkStatus, 5000)
      return () => clearInterval(interval)
    }
  }, [status.call_id])

  const startAgent = async () => {
    // 驗證名稱
    const trimmedName = userName.trim()
    if (!trimmedName) {
      setNameError('請輸入您的名稱')
      return
    }
    if (trimmedName.length < 2) {
      setNameError('名稱至少需要 2 個字')
      return
    }
    if (trimmedName.length > 20) {
      setNameError('名稱不能超過 20 個字')
      return
    }

    setNameError('')
    setLoading(true)
    try {
      const res = await fetch('/api/start', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          model: 'gemini',  // 統一使用 Gemini（支援視訊）
          example: 'custom',  // 固定使用自訂 Agent
          user_name: trimmedName  // 傳送用戶名稱
        }),
      })
      const data = await res.json()

      if (data.success) {
        // 直接跳轉到視訊通話
        window.open(data.demo_url, '_blank')
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
    if (!status.call_id) return

    try {
      await fetch('/api/stop', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ call_id: status.call_id }),
      })
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
              <h3>
                <Bot size={24} aria-hidden="true" />
                繁體中文 AI 語音助理
              </h3>
              <p className="example-description">
                Gemini 2.5 Flash Realtime - 支援視訊、語音、文字與天氣查詢
              </p>
            </div>

            <motion.div
              className="name-input-container"
              initial={{ opacity: 0, y: -10 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.1 }}
            >
              <label className="name-label">
                <span className="label-text">
                  <User size={16} aria-hidden="true" />
                  您的名稱
                </span>
                <motion.input
                  type="text"
                  className={`name-input ${nameError ? 'error' : ''}`}
                  placeholder="請輸入您的名稱..."
                  value={userName}
                  onChange={(e) => {
                    setUserName(e.target.value)
                    setNameError('')
                  }}
                  maxLength={20}
                  disabled={loading}
                  whileFocus={{ scale: 1.01 }}
                />
              </label>
              {nameError && (
                <motion.p
                  className="name-error"
                  role="alert"
                  aria-live="polite"
                  initial={{ opacity: 0, height: 0 }}
                  animate={{ opacity: 1, height: 'auto' }}
                  exit={{ opacity: 0, height: 0 }}
                >
                  <AlertCircle size={16} aria-hidden="true" />
                  {nameError}
                </motion.p>
              )}
              {userName && !nameError && (
                <motion.p
                  className="name-hint"
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                >
                  <CheckCircle size={16} aria-hidden="true" />
                  這個名稱會顯示在視訊通話中
                </motion.p>
              )}
            </motion.div>

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
              <h3>
                <Video size={20} aria-hidden="true" />
                視訊通話已開啟
              </h3>
              <p>
                視訊通話已在新視窗開啟。如果沒有自動開啟，請點擊下方按鈕。
              </p>
              <motion.button
                className="btn btn-primary"
                onClick={openDemo}
                whileHover={{ scale: 1.02 }}
                whileTap={{ scale: 0.98 }}
              >
                <Video size={20} />
                重新開啟視訊通話
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
        <h3>
          <Lightbulb size={20} aria-hidden="true" />
          使用說明
        </h3>
        <ol>
          <li>點擊「啟動 Agent」開始</li>
          <li>等待 Agent 準備完成</li>
          <li>點擊「開啟視訊通話」進入通話介面</li>
          <li>在瀏覽器中允許麥克風和攝影機權限</li>
          <li>開始與 AI 對話（支援語音和文字輸入）</li>
        </ol>
      </div>
    </motion.div>
  )
}

export default App
