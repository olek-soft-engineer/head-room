"""Headroom integrations with popular LLM frameworks.

Available integrations:
- LangChain: HeadroomChatModel, HeadroomCallbackHandler, optimize_messages
- MCP: HeadroomMCPCompressor, compress_tool_result, HeadroomMCPClientWrapper

Install LangChain support: pip install headroom[langchain]
"""

from .langchain import (
    HeadroomCallbackHandler,
    HeadroomChatModel,
    HeadroomRunnable,
    optimize_messages,
)
from .mcp import (
    DEFAULT_MCP_PROFILES,
    HeadroomMCPClientWrapper,
    HeadroomMCPCompressor,
    MCPCompressionResult,
    MCPToolProfile,
    compress_tool_result,
    compress_tool_result_with_metrics,
    create_headroom_mcp_proxy,
)

__all__ = [
    # LangChain
    "HeadroomChatModel",
    "HeadroomCallbackHandler",
    "optimize_messages",
    "HeadroomRunnable",
    # MCP
    "HeadroomMCPCompressor",
    "HeadroomMCPClientWrapper",
    "MCPCompressionResult",
    "MCPToolProfile",
    "compress_tool_result",
    "compress_tool_result_with_metrics",
    "create_headroom_mcp_proxy",
    "DEFAULT_MCP_PROFILES",
]
