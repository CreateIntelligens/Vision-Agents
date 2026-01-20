import { useEffect, useState } from 'react'
import {
  StreamVideo,
  StreamVideoClient,
  StreamCall,
  CallControls,
  SpeakerLayout,
  CallParticipantsList,
} from '@stream-io/video-react-sdk'
import '@stream-io/video-react-sdk/dist/css/styles.css'

interface VideoCallProps {
  apiKey: string
  token: string
  userId: string
  callId: string
  onClose: () => void
}

export function VideoCall({ apiKey, token, userId, callId, onClose }: VideoCallProps) {
  const [client, setClient] = useState<StreamVideoClient | null>(null)
  const [call, setCall] = useState<StreamCall | null>(null)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    const initClient = async () => {
      try {
        const videoClient = StreamVideoClient.getOrCreateInstance({
          apiKey,
          user: { id: userId },
          token,
        })

        const videoCall = videoClient.call('default', callId)
        await videoCall.getOrCreate()
        await videoCall.join()

        setClient(videoClient)
        setCall(videoCall)
      } catch (err) {
        console.error('❌ Failed to join call:', err)
        setError(err instanceof Error ? err.message : String(err))
      }
    }

    initClient()

    return () => {
      call?.leave()
    }
  }, [apiKey, token, userId, callId])

  if (error) {
    return (
      <div style={{ 
        position: 'fixed', 
        top: 0, 
        left: 0, 
        right: 0, 
        bottom: 0, 
        background: '#000',
        color: '#ef4444',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        flexDirection: 'column',
        gap: '1rem',
        zIndex: 9999 
      }}>
        <div style={{ fontSize: '3rem' }}>❌</div>
        <h2>連接失敗</h2>
        <p>{error}</p>
        <button 
          onClick={onClose}
          style={{
            padding: '0.75rem 2rem',
            background: '#ef4444',
            color: 'white',
            border: 'none',
            borderRadius: '0.5rem',
            cursor: 'pointer',
            fontSize: '1rem',
            fontWeight: 'bold'
          }}
        >
          返回
        </button>
      </div>
    )
  }

  if (!client || !call) {
    return (
      <div style={{ 
        position: 'fixed', 
        top: 0, 
        left: 0, 
        right: 0, 
        bottom: 0, 
        background: '#000',
        color: '#fff',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        flexDirection: 'column',
        gap: '1rem',
        zIndex: 9999 
      }}>
        <div className="spinner" style={{
          width: '48px',
          height: '48px',
          border: '4px solid rgba(255,255,255,0.1)',
          borderTopColor: '#00d9ff',
          borderRadius: '50%',
          animation: 'spin 1s linear infinite'
        }} />
        <p style={{ fontSize: '1.2rem' }}>正在連接視訊通話...</p>
        <p style={{ fontSize: '0.9rem', color: '#888' }}>Call ID: {callId}</p>
      </div>
    )
  }

  return (
    <StreamVideo client={client}>
      <StreamCall call={call}>
        <div style={{ 
          position: 'fixed', 
          top: 0, 
          left: 0, 
          right: 0, 
          bottom: 0, 
          background: '#000',
          display: 'flex',
          flexDirection: 'column'
        }}>
          {/* 頂部資訊 */}
          <div style={{
            padding: '1rem',
            background: 'rgba(0,0,0,0.8)',
            color: '#fff',
            display: 'flex',
            justifyContent: 'space-between',
            alignItems: 'center'
          }}>
            <div>
              <h3 style={{ margin: 0, fontSize: '1.2rem' }}>🤖 Vision Agent</h3>
              <p style={{ margin: 0, fontSize: '0.9rem', color: '#888' }}>Call ID: {callId}</p>
            </div>
            <button 
              onClick={onClose}
              style={{
                padding: '0.5rem 1.5rem',
                background: '#ef4444',
                color: 'white',
                border: 'none',
                borderRadius: '0.5rem',
                cursor: 'pointer',
                fontSize: '1rem',
                fontWeight: 'bold'
              }}
            >
              ❌ 離開通話
            </button>
          </div>

          {/* 主要視訊區域 - 使用 Stream 的 SpeakerLayout */}
          <div style={{ flex: 1, position: 'relative' }}>
            <SpeakerLayout />
          </div>

          {/* 底部控制列 */}
          <div style={{
            padding: '1rem',
            background: 'rgba(0,0,0,0.9)',
          }}>
            <CallControls />
          </div>

          {/* 側邊參與者列表 */}
          <div style={{
            position: 'absolute',
            top: '80px',
            right: '1rem',
            width: '250px',
            maxHeight: 'calc(100% - 180px)',
            background: 'rgba(0,0,0,0.8)',
            borderRadius: '0.5rem',
            overflow: 'auto',
            padding: '0.5rem'
          }}>
            <CallParticipantsList />
          </div>
        </div>
      </StreamCall>
    </StreamVideo>
  )
}
