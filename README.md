---
title: Itihas
emoji: ⚔️
colorFrom: yellow
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
---


# Itihas

Multi-agent adversarial AI system for Indian military and political history, 1600–1947.

## Environment variables (set in Space Settings → Variables and secrets)

| Variable | Required | Secret? | Description |
|---|---|---|---|
| `HF_API_TOKEN` | Yes | Yes | HuggingFace API token for Inference API calls |
| `DATABASE_URL` | Yes | Yes | Neon Tech PostgreSQL connection string |
| `HF_MODEL` | No | No | Override default LLM model |

Do not commit secrets to the repo. Set them in the Space's Settings tab.