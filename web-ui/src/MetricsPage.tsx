import { Activity } from 'lucide-react'
import { MetricsDashboard } from './MetricsDashboard'
import './MetricsPage.css'
import './index.css'

export function MetricsPage() {
  return (
    <div className="metrics-page">
      <div className="metrics-page-header">
        <h1>
          <Activity size={32} />
          Vision Agents 性能監控
        </h1>
        <p className="subtitle">即時系統指標與性能分析</p>
      </div>
      
      <MetricsDashboard callId="standalone" />
      
      <div className="metrics-page-footer">
        <div className="footer-links">
          <a href="/api/metrics/json" target="_blank" rel="noopener noreferrer">
            JSON API
          </a>
          <span>•</span>
          <a href="/api/metrics/prometheus" target="_blank" rel="noopener noreferrer">
            Prometheus 格式
          </a>
          <span>•</span>
          <a href="/" className="back-home">
            返回主頁
          </a>
        </div>
        <p className="copyright">
          Vision Agents © 2026 - Powered by Prometheus & OpenTelemetry
        </p>
      </div>
    </div>
  )
}
