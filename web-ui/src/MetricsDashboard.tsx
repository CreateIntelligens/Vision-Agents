import { useState, useEffect } from 'react'
import { motion } from 'framer-motion'
import { Activity, Zap, MessageSquare, Clock, TrendingUp, RefreshCw } from 'lucide-react'
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts'

interface MetricsData {
  timestamp: number
  llm_latency?: number
  llm_tokens_input?: number
  llm_tokens_output?: number
  realtime_sessions?: number
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

        // 提取指標數值
        if (data.llm_latency_ms?.samples) {
          const latencySample = data.llm_latency_ms.samples.find((s: any) => s.name.includes('_sum'))
          const countSample = data.llm_latency_ms.samples.find((s: any) => s.name.includes('_count'))
          if (latencySample && countSample && countSample.value > 0) {
            newPoint.llm_latency = latencySample.value / countSample.value
          }
        }

        if (data.llm_tokens_input?.samples?.[0]) {
          newPoint.llm_tokens_input = data.llm_tokens_input.samples[0].value
        }

        if (data.llm_tokens_output?.samples?.[0]) {
          newPoint.llm_tokens_output = data.llm_tokens_output.samples[0].value
        }

        if (data.realtime_sessions?.samples?.[0]) {
          newPoint.realtime_sessions = data.realtime_sessions.samples[0].value
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
    if (!metrics || !metrics[metricName]) return 'N/A'
    const samples = metrics[metricName].samples
    if (!samples || samples.length === 0) return '0'

    // 如果是 counter，返回總數
    if (metricName.includes('tokens') || metricName.includes('sessions')) {
      return Math.round(samples[0].value).toString()
    }

    // 如果是 histogram，計算平均值
    const sumSample = samples.find((s: any) => s.name.includes('_sum'))
    const countSample = samples.find((s: any) => s.name.includes('_count'))
    if (sumSample && countSample && countSample.value > 0) {
      return Math.round(sumSample.value / countSample.value).toString()
    }

    return samples[0].value.toFixed(2)
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
        {/* LLM 延遲 */}
        <motion.div
          className="metric-card"
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.1 }}
        >
          <div className="metric-header">
            <Clock size={24} />
            <h3>LLM 平均延遲</h3>
          </div>
          <div className="metric-value">
            {getMetricValue('llm_latency_ms')} <span className="unit">ms</span>
          </div>
        </motion.div>

        {/* Token 輸入 */}
        <motion.div
          className="metric-card"
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.2 }}
        >
          <div className="metric-header">
            <MessageSquare size={24} />
            <h3>輸入 Tokens</h3>
          </div>
          <div className="metric-value">
            {getMetricValue('llm_tokens_input')} <span className="unit">tokens</span>
          </div>
        </motion.div>

        {/* Token 輸出 */}
        <motion.div
          className="metric-card"
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.3 }}
        >
          <div className="metric-header">
            <Zap size={24} />
            <h3>輸出 Tokens</h3>
          </div>
          <div className="metric-value">
            {getMetricValue('llm_tokens_output')} <span className="unit">tokens</span>
          </div>
        </motion.div>

        {/* Realtime Sessions */}
        <motion.div
          className="metric-card"
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.4 }}
        >
          <div className="metric-header">
            <Activity size={24} />
            <h3>Realtime 連線</h3>
          </div>
          <div className="metric-value">
            {getMetricValue('realtime_sessions')} <span className="unit">sessions</span>
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
          <h3><TrendingUp size={20} /> 延遲趨勢</h3>
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
                dataKey="llm_latency"
                stroke="#00d9ff"
                name="LLM 延遲 (ms)"
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
