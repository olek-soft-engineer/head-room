# Headroom Documentation

Welcome to the Headroom documentation.

## Quick Links

- [Getting Started](getting-started.md)
- [Proxy Server](proxy.md)
- [Transforms](transforms.md)
- [API Reference](api.md)
- [Architecture](ARCHITECTURE.md)

## Overview

Headroom is the Context Optimization Layer for LLM applications. It reduces your LLM costs by 50-90% through intelligent context compression.

### Core Concepts

1. **Transforms**: Stateless functions that modify message arrays to reduce tokens
2. **Providers**: Adapters for different LLM providers (OpenAI, Anthropic, etc.)
3. **Pipeline**: Chains multiple transforms together
4. **Proxy**: HTTP server that applies transforms transparently

### Getting Help

- [GitHub Issues](https://github.com/headroom-sdk/headroom/issues) - Bug reports
- [GitHub Discussions](https://github.com/headroom-sdk/headroom/discussions) - Questions
- [Discord](https://discord.gg/headroom) - Community chat
