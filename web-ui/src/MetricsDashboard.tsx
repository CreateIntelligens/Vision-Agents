import { useState, useEffect } from 'react'
import { motion } from 'framer-motion'
import { Activity, Zap, MessageSquare, Clock, TrendingUp, RefreshCw } from 'lucide-react'
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts'
import './MetricsDashboard.css'

interface MetricsData {
  timestamp: number
  audio_latency?: number      // 音訊輸出延遲
  ai_transcriptions?: number  // AI 語音輸出次數
  user_transcriptions?: number // 用戶語音輸入次數
}

interface MetricsDashboardProps {
  callId: string
}

export function MetricsDashboard({ callId }: MetricsDashboardProps) {
  const [metrics, setMetrics] = useState<any>(null)
  const [history, setHistory] = useState<MetricsData[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    const fetchMetrics = async () => {
      try {
        const res = await fetch('/api/metrics/json')
        if (!res.ok) throw new Error('Failed to fetch metrics')
        const data = await res.json()
        setMetrics(data)
        setError(null)

        // 添加到歷史記錄
        const now = Date.now()
        const newPoint: MetricsData = {
          timestamp: now,
        }

        // 音訊輸出延遲（histogram，需要計算平均值）
        const audioLatencyMetric = data.realtime_audio_output_duration_ms_milliseconds
        if (audioLatencyMetric?.samples) {
          const sumSample = audioLatencyMetric.samples.find((s: any) => s.name.includes('_sum'))
          const countSample = audioLatencyMetric.samples.find((s: any) => s.name.includes('_count'))
          if (sumSample && countSample && countSample.value > 0) {
            newPoint.audio_latency = sumSample.value / countSample.value
          }
        }

        // AI 語音輸出次數
        if (data.realtime_transcriptions_agent?.samples?.[0]) {
          newPoint.ai_transcriptions = data.realtime_transcriptions_agent.samples[0].value
        }

        // 用戶語音輸入次數
        if (data.realtime_transcriptions_user?.samples?.[0]) {
          newPoint.user_transcriptions = data.realtime_transcriptions_user.samples[0].value
        }

        setHistory(prev => [...prev.slice(-19), newPoint]) // 保留最近 20 個點
        setLoading(false)
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Unknown error')
        setLoading(false)
      }
    }

    fetchMetrics()
    const interval = setInterval(fetchMetrics, 2000) // 每 2 秒更新

    return () => clearInterval(interval)
  }, [])

  const getMetricValue = (metricName: string): string => {
    if (!metrics) return 'N/A'

    // 支持多種指標名稱（實際名稱 -> 別名）
    const metricMap: Record<string, string[]> = {
      'llm_tool_latency_ms': ['llm_tool_latency_ms_milliseconds', 'llm_latency_ms'],
      'llm_tool_calls': ['llm_tool_calls', 'llm_tool_calls_total'],
      'realtime_transcriptions_agent': ['realtime_transcriptions_agent', 'realtime_transcriptions_agent_total'],
      'realtime_transcriptions_user': ['realtime_transcriptions_user'],
      'realtime_audio_output_bytes': ['realtime_audio_output_bytes'],
      'process_memory_mb': ['process_resident_memory_bytes'],
    }

    let metricData = metrics[metricName]
    if (!metricData && metricMap[metricName]) {
      for (const alt of metricMap[metricName]) {
        if (metrics[alt]) {
          metricData = metrics[alt]
          break
        }
      }
    }

    if (!metricData) return 'N/A'
    const samples = metricData.samples
    if (!samples || samples.length === 0) return '0'

    // 特殊處理：記憶體轉 MB
    if (metricName === 'process_memory_mb') {
      return Math.round(samples[0].value / 1024 / 1024).toString()
    }

    // 特殊處理：音訊轉 MB
    if (metricName === 'realtime_audio_output_bytes') {
      return (samples[0].value / 1024 / 1024).toFixed(1)
    }

    // 如果是 histogram，計算平均值
    const sumSample = samples.find((s: any) => s.name.includes('_sum'))
    const countSample = samples.find((s: any) => s.name.includes('_count'))
    if (sumSample && countSample && countSample.value > 0) {
      return Math.round(sumSample.value / countSample.value).toString()
    }

    return Math.round(samples[0].value).toString()
  }

  if (loading) {
    return (
      <div className="metrics-dashboard loading">
        <RefreshCw className="spinning" size={32} />
        <p>正在載入監控數據...</p>
      </div>
    )
  }

  if (error) {
    return (
      <div className="metrics-dashboard error">
        <Activity size={32} />
        <p>無法載入監控數據</p>
        <code>{error}</code>
      </div>
    )
  }

  return (
    <div className="metrics-dashboard">
      <div className="metrics-grid">
        {/* AI 語音轉錄 */}
        <motion.div
          className="metric-card"
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.1 }}
        >
          <div className="metric-header">
            <MessageSquare size={24} />
            <h3>AI 語音輸出</h3>
          </div>
          <div className="metric-value">
            {getMetricValue('realtime_transcriptions_agent')} <span className="unit">次</span>
          </div>
        </motion.div>

        {/* 用戶語音轉錄 */}
        <motion.div
          className="metric-card"
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.2 }}
        >
          <div className="metric-header">
            <Activity size={24} />
            <h3>用戶語音輸入</h3>
          </div>
          <div className="metric-value">
            {getMetricValue('realtime_transcriptions_user')} <span className="unit">次</span>
          </div>
        </motion.div>

        {/* 音訊輸出大小 */}
        <motion.div
          className="metric-card"
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.3 }}
        >
          <div className="metric-header">
            <Zap size={24} />
            <h3>音訊輸出</h3>
          </div>
          <div className="metric-value">
            {getMetricValue('realtime_audio_output_bytes')} <span className="unit">MB</span>
          </div>
        </motion.div>

        {/* 記憶體使用 */}
        <motion.div
          className="metric-card"
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.4 }}
        >
          <div className="metric-header">
            <Clock size={24} />
            <h3>記憶體使用</h3>
          </div>
          <div className="metric-value">
            {getMetricValue('process_memory_mb')} <span className="unit">MB</span>
          </div>
        </motion.div>
      </div>

      {/* 歷史圖表 */}
      {history.length > 1 && (
        <motion.div
          className="metrics-chart"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 0.5 }}
        >
          <h3><TrendingUp size={20} /> 對話趨勢</h3>
          <ResponsiveContainer width="100%" height={200}>
            <LineChart data={history}>
              <CartesianGrid strokeDasharray="3 3" stroke="#333" />
              <XAxis
                dataKey="timestamp"
                tickFormatter={(ts) => new Date(ts).toLocaleTimeString()}
                stroke="#666"
              />
              <YAxis stroke="#666" />
              <Tooltip
                contentStyle={{ background: '#1a1a1a', border: '1px solid #333' }}
                labelFormatter={(ts) => new Date(ts).toLocaleTimeString()}
              />
              <Legend />
              <Line
                type="monotone"
                dataKey="ai_transcriptions"
                stroke="#00d9ff"
                name="AI 語音輸出"
                dot={false}
              />
              <Line
                type="monotone"
                dataKey="user_transcriptions"
                stroke="#ff6b6b"
                name="用戶語音輸入"
                dot={false}
              />
            </LineChart>
          </ResponsiveContainer>
        </motion.div>
      )}

      <div className="metrics-info">
        <p>
          <Activity size={14} />
          <span>監控數據每 2 秒更新一次</span>
        </p>
        <p>
          <span>CALL ID: <code>{callId}</code></span>
        </p>
      </div>
    </div>
  )
}
