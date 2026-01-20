import React from 'react'
import ReactDOM from 'react-dom/client'
import { VideoCall } from './VideoCall'

// 從 URL 參數讀取 token, apiKey, userId, callId
const params = new URLSearchParams(window.location.search)
const apiKey = params.get('apiKey')
const token = params.get('token')
const userId = params.get('userId')
const callId = params.get('callId')

const handleClose = () => {
  // 通知後端停止 Agent
  if (callId) {
    fetch(`/api/agent/${callId}/stop`, { method: 'POST' })
      .catch(err => console.error('❌ Failed to stop agent:', err))
  }
  window.close()
}

// 監聽視窗關閉事件
window.addEventListener('beforeunload', () => {
  if (callId) {
    // 同步請求確保在關閉前完成
    navigator.sendBeacon(`/api/agent/${callId}/stop`)
  }
})

if (!apiKey || !token || !userId || !callId) {
  document.body.innerHTML = `
    <div style="display: flex; align-items: center; justify-content: center; height: 100vh; background: #000; color: #ef4444; flex-direction: column; gap: 1rem;">
      <h2>❌ 參數錯誤</h2>
      <p>缺少必要的連接參數</p>
      <button onclick="window.close()" style="padding: 0.75rem 2rem; background: #ef4444; color: white; border: none; border-radius: 0.5rem; cursor: pointer;">
        關閉
      </button>
    </div>
  `
} else {
  ReactDOM.createRoot(document.getElementById('call-root')!).render(
    <React.StrictMode>
      <VideoCall
        apiKey={apiKey}
        token={token}
        userId={userId}
        callId={callId}
        onClose={handleClose}
      />
    </React.StrictMode>
  )
}
