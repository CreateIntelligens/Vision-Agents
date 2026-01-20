import { useState, useEffect } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import {
  Video, Loader2, Power, PowerOff, Bot, User,
  Camera, BarChart3, Zap, Shield, Activity, ChevronRight,
  CheckCircle2, AlertCircle, Terminal, Lock, Unlock
} from 'lucide-react'
import { MetricsDashboard } from './MetricsDashboard'
import './App.css'

type AgentStatus = {
  running: boolean
  call_id: string | null
  model: string | null
}

type AgentMode = 'custom' | 'security_camera' | 'prometheus_metrics'

interface AgentConfig {
  id: AgentMode
  name: string
  code: string
  description: string
  icon: React.ReactNode
  color: string
  features: string[]
}

const AGENT_CONFIGS: AgentConfig[] = [
  {
    id: 'custom',
    name: '通用助理',
    code: 'ASSIST-01',
    description: '多功能 AI 助理｜視訊分析、知識庫檢索、即時搜尋',
    icon: <Bot size={48} />,
    color: '#00d9ff',
    features: ['VIDEO_ANALYSIS', 'RAG_SEARCH', 'REAL_TIME_WEB', 'VOICE_AI']
  },
  {
    id: 'security_camera',
    name: '安保監控',
    code: 'SECURE-02',
    description: '智能監控系統｜人臉辨識、物體檢測、竊盜預警',
    icon: <Camera size={48} />,
    color: '#00ffc8',
    features: ['FACE_DETECT', 'YOLO_TRACK', 'THEFT_ALERT', 'AUTO_REPORT']
  },
  {
    id: 'prometheus_metrics',
    name: '性能監控',
    code: 'METRICS-03',
    description: '系統性能分析｜OpenTelemetry 指標、即時監控',
    icon: <BarChart3 size={48} />,
    color: '#b84dff',
    features: ['OPENTELEMETRY', 'PROMETHEUS', 'LIVE_METRICS', 'PERFORMANCE']
  }
]

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
  const [selectedMode, setSelectedMode] = useState<AgentMode>('custom')
  const [systemTime, setSystemTime] = useState(new Date())

  // System time ticker
  useEffect(() => {
    const timer = setInterval(() => setSystemTime(new Date()), 1000)
    return () => clearInterval(timer)
  }, [])

  // Status polling
  useEffect(() => {
    const checkStatus = async () => {
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
        console.error('Status check failed:', error)
      }
    }

    if (status.call_id) {
      checkStatus()
      const interval = setInterval(checkStatus, 5000)
      return () => clearInterval(interval)
    }
  }, [status.call_id])

  // 生成隨機名稱
  const generateRandomName = () => {
    const adjectives = ['快樂', '聰明', '勇敢', '可愛', '神秘', '活潑', '溫柔', '帥氣']
    const animals = ['小貓', '小狗', '兔子', '熊貓', '企鵝', '狐狸', '老虎', '龍']
    const adj = adjectives[Math.floor(Math.random() * adjectives.length)]
    const animal = animals[Math.floor(Math.random() * animals.length)]
    const num = Math.floor(Math.random() * 100)
    return `${adj}${animal}${num}`
  }

  const startAgent = async () => {
    const trimmedName = userName.trim()

    // 如果有輸入名稱，檢查長度
    if (trimmedName && trimmedName.length < 2) {
      setNameError('名稱至少需要 2 個字')
      return
    }
    if (trimmedName.length > 20) {
      setNameError('名稱不能超過 20 個字')
      return
    }

    // 沒有輸入就用隨機名稱
    const finalName = trimmedName || generateRandomName()
    
    // 更新顯示的用戶名稱（重要！讓 UI 顯示實際使用的名稱）
    setUserName(finalName)

    setNameError('')
    setLoading(true)

    try {
      const res = await fetch('/api/start', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          model: 'gemini',
          example: selectedMode,
          user_name: finalName
        }),
      })
      const data = await res.json()

      if (data.success) {
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
      console.error('Stop failed:', error)
    }
  }

  const openDemo = () => {
    if (demoUrl) window.open(demoUrl, '_blank')
  }

  const selectedConfig = AGENT_CONFIGS.find(c => c.id === selectedMode)!

  return (
    <div className="tech-noir-container">
      {/* Animated background */}
      <div className="bg-grid" />
      <div className="bg-gradient" />
      <div className="scanlines" />
      
      {/* Data stream effect */}
      <div className="data-stream">
        {Array.from({ length: 20 }).map((_, i) => (
          <div key={i} className="data-bit" style={{ left: `${i * 5}%`, animationDelay: `${i * 0.3}s` }}>
            {Math.random() > 0.5 ? '1' : '0'}
          </div>
        ))}
      </div>

      <div className="content-container">
        {/* Header */}
        <motion.header
          className="system-header"
          initial={{ y: -50, opacity: 0 }}
          animate={{ y: 0, opacity: 1 }}
          transition={{ duration: 0.8, ease: [0.6, 0.05, 0.01, 0.9] }}
        >
          <div className="header-main">
            <div className="logo-block">
              <Terminal className="logo-icon" />
              <div className="logo-text-group">
                <h1 className="logo-title">VISION AGENTS</h1>
                <p className="logo-subtitle">CLASSIFIED SYSTEM ACCESS</p>
              </div>
            </div>

            <div className="system-info">
              <div className="info-item">
                <span className="info-label">SYSTEM TIME</span>
                <span className="info-value">{systemTime.toLocaleTimeString('en-US', { hour12: false })}</span>
              </div>
              <div className="info-item">
                <span className="info-label">STATUS</span>
                <motion.div 
                  className={`status-indicator ${status.running ? 'active' : 'standby'}`}
                  animate={{ opacity: status.running ? [1, 0.5, 1] : 1 }}
                  transition={{ repeat: status.running ? Infinity : 0, duration: 2 }}
                >
                  <Activity size={14} />
                  <span>{status.running ? 'ACTIVE' : 'STANDBY'}</span>
                </motion.div>
              </div>
            </div>
          </div>

          <div className="security-bar">
            <Lock size={12} />
            <span>SECURITY CLEARANCE: LEVEL 5</span>
            <div className="bar-fill" />
          </div>
        </motion.header>

        <AnimatePresence mode="wait">
          {!status.running ? (
            <motion.main
              key="setup"
              className="setup-grid"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              transition={{ duration: 0.5 }}
            >
              {/* Agent Selection */}
              <motion.section
                className="agent-selection"
                initial={{ x: -50, opacity: 0 }}
                animate={{ x: 0, opacity: 1 }}
                transition={{ delay: 0.2, duration: 0.6 }}
              >
                <div className="section-header">
                  <Shield size={18} />
                  <h2>AGENT MODE SELECTION</h2>
                  <div className="header-line" />
                </div>

                <div className="agent-grid">
                  {AGENT_CONFIGS.map((config, index) => (
                    <motion.button
                      key={config.id}
                      className={`agent-card ${selectedMode === config.id ? 'selected' : ''}`}
                      onClick={() => setSelectedMode(config.id)}
                      disabled={loading}
                      initial={{ opacity: 0, y: 20 }}
                      animate={{ opacity: 1, y: 0 }}
                      transition={{ delay: 0.4 + index * 0.15 }}
                      whileHover={{ scale: 1.02, y: -4 }}
                      whileTap={{ scale: 0.98 }}
                      style={{ '--card-color': config.color } as React.CSSProperties}
                    >
                      <div className="card-corner tl" />
                      <div className="card-corner tr" />
                      <div className="card-corner bl" />
                      <div className="card-corner br" />

                      <div className="card-header">
                        <div className="card-icon">{config.icon}</div>
                        <div className="card-code">{config.code}</div>
                      </div>

                      <div className="card-body">
                        <h3 className="card-title">{config.name}</h3>
                        <p className="card-desc">{config.description}</p>

                        <div className="card-features">
                          {config.features.map((feature, i) => (
                            <span key={i} className="feature-tag">
                              <Zap size={10} />
                              {feature}
                            </span>
                          ))}
                        </div>
                      </div>

                      {selectedMode === config.id && (
                        <motion.div
                          className="selected-badge"
                          initial={{ scale: 0 }}
                          animate={{ scale: 1 }}
                          transition={{ type: 'spring', stiffness: 300 }}
                        >
                          <CheckCircle2 size={20} />
                        </motion.div>
                      )}
                    </motion.button>
                  ))}
                </div>
              </motion.section>

              {/* Control Panel */}
              <motion.section
                className="control-panel"
                initial={{ x: 50, opacity: 0 }}
                animate={{ x: 0, opacity: 1 }}
                transition={{ delay: 0.3, duration: 0.6 }}
              >
                <div className="panel-box">
                  <div className="panel-header">
                    <Terminal size={16} />
                    <span>OPERATOR CREDENTIALS</span>
                  </div>

                  <div className="credential-input">
                    <label className="input-label">
                      <User size={14} />
                      <span>OPERATOR ID</span>
                    </label>
                    <motion.input
                      type="text"
                      className={`cyber-input ${nameError ? 'error' : ''}`}
                      placeholder="選填，留空則隨機生成..."
                      value={userName}
                      onChange={(e) => {
                        setUserName(e.target.value)
                        setNameError('')
                      }}
                      maxLength={20}
                      disabled={loading}
                      whileFocus={{ scale: 1.01 }}
                    />
                    
                    {nameError && (
                      <motion.div
                        className="input-error"
                        initial={{ opacity: 0, height: 0 }}
                        animate={{ opacity: 1, height: 'auto' }}
                      >
                        <AlertCircle size={12} />
                        {nameError}
                      </motion.div>
                    )}

                    {userName && !nameError && (
                      <motion.div
                        className="input-success"
                        initial={{ opacity: 0 }}
                        animate={{ opacity: 1 }}
                      >
                        <CheckCircle2 size={12} />
                        CREDENTIALS VERIFIED
                      </motion.div>
                    )}
                  </div>

                  <div className="selected-agent-info">
                    <div className="info-row">
                      <span className="info-key">SELECTED MODE</span>
                      <span className="info-val" style={{ color: selectedConfig.color }}>
                        {selectedConfig.code}
                      </span>
                    </div>
                    <div className="info-row">
                      <span className="info-key">AGENT NAME</span>
                      <span className="info-val">{selectedConfig.name}</span>
                    </div>
                    <div className="info-row">
                      <span className="info-key">CLEARANCE</span>
                      <span className="info-val">AUTHORIZED</span>
                    </div>
                  </div>

                  <motion.button
                    className="launch-btn"
                    onClick={startAgent}
                    disabled={loading || !!nameError}
                    whileHover={{ scale: 1.02 }}
                    whileTap={{ scale: 0.98 }}
                    style={{ background: selectedConfig.color }}
                  >
                    {loading ? (
                      <>
                        <Loader2 className="spinning" size={18} />
                        INITIALIZING SYSTEM...
                      </>
                    ) : (
                      <>
                        <Power size={18} />
                        ENGAGE {selectedConfig.code}
                        <ChevronRight size={18} />
                      </>
                    )}
                  </motion.button>
                </div>
              </motion.section>
            </motion.main>
          ) : (
            <motion.main
              key="active"
              className="active-session"
              initial={{ opacity: 0, scale: 0.95 }}
              animate={{ opacity: 1, scale: 1 }}
              exit={{ opacity: 0, scale: 0.95 }}
            >
              <div className="session-panel">
                <div className="session-header">
                  <div 
                    className="session-icon"
                    style={{ background: selectedConfig.color }}
                  >
                    {selectedConfig.icon}
                  </div>
                  <div className="session-info">
                    <h2>{selectedConfig.name} SESSION ACTIVE</h2>
                    <p className="session-detail">
                      <span>CALL ID:</span> <code>{status.call_id}</code>
                    </p>
                    <p className="session-detail">
                      <span>MODEL:</span> <code>{status.model}</code>
                    </p>
                    <p className="session-detail">
                      <span>OPERATOR:</span> <code>{userName}</code>
                    </p>
                  </div>
                </div>

                {/* 根據 agent 類型顯示不同的內容 */}
                {selectedMode === 'prometheus_metrics' ? (
                  <div className="split-screen-container">
                    {/* 左側：視訊通話 */}
                    <div className="split-video">
                      <div className="video-header">
                        <Video size={16} />
                        <span>語音通話界面</span>
                      </div>
                      {demoUrl && (
                        <iframe
                          src={demoUrl}
                          className="video-iframe"
                          allow="camera; microphone; fullscreen; display-capture"
                        />
                      )}
                      <div className="video-footer">
                        <p>與 AI 對話，右側即時顯示性能指標</p>
                      </div>
                    </div>

                    {/* 右側：Metrics 儀表板 */}
                    <div className="split-metrics">
                      <div className="metrics-header">
                        <Activity size={16} />
                        <span>即時性能監控</span>
                      </div>
                      <MetricsDashboard callId={status.call_id || ''} />
                    </div>

                    {/* 底部操作按鈕 */}
                    <div className="split-actions">
                      <motion.button
                        className="session-btn danger"
                        onClick={stopAgent}
                        whileHover={{ scale: 1.05 }}
                        whileTap={{ scale: 0.95 }}
                      >
                        <PowerOff size={18} />
                        TERMINATE SESSION
                      </motion.button>
                    </div>
                  </div>
                ) : (
                  <div className="session-actions">
                    <motion.button
                      className="session-btn primary"
                      onClick={openDemo}
                      whileHover={{ scale: 1.05 }}
                      whileTap={{ scale: 0.95 }}
                    >
                      <Video size={18} />
                      OPEN VIDEO LINK
                    </motion.button>

                    <motion.button
                      className="session-btn danger"
                      onClick={stopAgent}
                      whileHover={{ scale: 1.05 }}
                      whileTap={{ scale: 0.95 }}
                    >
                      <PowerOff size={18} />
                      TERMINATE SESSION
                    </motion.button>
                  </div>
                )}
              </div>
            </motion.main>
          )}
        </AnimatePresence>

        {/* Footer */}
        <motion.footer
          className="system-footer"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 1 }}
        >
          <Unlock size={12} />
          <span>AUTHORIZED PERSONNEL ONLY</span>
          <span className="separator">|</span>
          <span>GEMINI 2.5 FLASH REALTIME</span>
          <span className="separator">|</span>
          <span>SYSTEM BUILD 2026.01</span>
        </motion.footer>
      </div>
    </div>
  )
}

export default App
