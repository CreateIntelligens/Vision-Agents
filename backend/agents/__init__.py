"""Backend Agents Package

Three agent modes:
- custom: General-purpose AI assistant
- security_camera: Intelligent security monitoring
- prometheus_metrics: Performance monitoring with OpenTelemetry
"""

from .custom import create_agent as create_custom_agent
from .security_camera import create_agent as create_security_agent
from .prometheus_metrics import create_agent as create_metrics_agent

# 向後相容：AGENT_TYPES 字典供 app.py 使用
AGENT_TYPES = {
    "custom": create_custom_agent,
    "security_camera": create_security_agent,
    "prometheus_metrics": create_metrics_agent,
}

__all__ = [
    'create_custom_agent',
    'create_security_agent',
    'create_metrics_agent',
    'AGENT_TYPES',
]
