# Backend Agents

Three agent modes for Vision Agents:

## 1. Custom Agent (`custom/`)
- General-purpose AI assistant
- Video analysis, RAG search, real-time web
- Entry point: `create_agent(call_id, user_name)`

## 2. Security Camera (`security_camera/`)
- Intelligent security monitoring
- Face detection, YOLO tracking, theft alerts  
- Entry point: `create_agent(call_id, user_name)`

## 3. Prometheus Metrics (`prometheus_metrics/`)
- Performance monitoring
- OpenTelemetry + Prometheus integration
- Entry point: `create_agent(call_id, user_name)`

## Usage

```python
from backend.agents import (
    create_custom_agent,
    create_security_agent,
    create_metrics_agent,
)

# Create an agent
agent = await create_custom_agent(call_id="test", user_name="User")
```

## Structure

```
backend/agents/
├── __init__.py (Package exports)
├── base.py (Shared ChatListenerProcessor)
├── custom/ (通用助理)
│   ├── __init__.py
│   └── agent.py
├── security_camera/ (安保監控)
│   ├── __init__.py
│   └── agent.py
└── prometheus_metrics/ (性能監控)
    ├── __init__.py
    └── agent.py
```
