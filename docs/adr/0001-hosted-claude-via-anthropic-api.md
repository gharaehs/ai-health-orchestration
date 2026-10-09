# Hosted Claude via the direct Anthropic API

All LLM calls go to Claude Opus 5.5 through the direct Anthropic API, replacing the self-hosted Llama 3.1 8B on vLLM (a GPU g4dn.xlarge) and the LoRA fine-tuning work from Modules 1–2. Self-hosting tied the system to an expensive GPU instance and a far weaker model; dropping it lets the stack run on a small CPU-only instance. We chose the Anthropic API over Amazon Bedrock for direct access to the newest models and features with less AWS setup, accepting an `ANTHROPIC_API_KEY` secret on the server instead of IAM-role auth.

## Consequences

- The local-LLM code (vLLM service, model download, `training/`) is removed; git history is its archive.
- No runtime provider switch: one shared client talks only to Anthropic.
- Cost is now per-token and exposed to anyone who can reach the app, so the security group is locked to the owner's IP and a spend limit is set in the Anthropic Console.
