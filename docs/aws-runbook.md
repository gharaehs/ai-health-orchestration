# Runbook: AWS migration — CPU EC2 + Claude Opus 5.5 via Bedrock

Moves the stack from the personal AWS account (GPU host, Llama 3.1 8B on vLLM) to the
company AWS account (CPU-only EC2, Claude Opus 5.5 through the Bedrock Converse API).
Plan: `plans/aws-migration.md`. These steps are executed manually by an operator.

## 0. Open prerequisites (confirm before starting)

| Item | Owner | Value |
|------|-------|-------|
| Bedrock region (`AWS_REGION`) | company cloud team | _TBD_ |
| Opus 5.5 model or inference-profile ID (`BEDROCK_MODEL_ID`) | company cloud team | _TBD_ (cross-region profiles look like `us.anthropic.…` / `eu.anthropic.…`) |
| Bedrock allowed + Opus 5.5 model access granted in that region | company cloud team | _TBD_ |
| Bedrock quota (requests/min, tokens/min) for Opus 5.5 | company cloud team | _TBD_ |
| Data classification of health profiles / blood markers; logging & retention rules | security / DPO | _TBD_ |
| Domain, TLS, SSO/WAF requirements | company cloud team | _TBD_ |

Do not go live until data classification and logging requirements are confirmed.

## 1. Discover the current deployment (personal account)

```bash
aws ec2 describe-instances --filters Name=tag:Name,Values=<name> \
  --query 'Reservations[].Instances[].{id:InstanceId,type:InstanceType,az:Placement.AvailabilityZone,sg:SecurityGroups,ip:PublicIpAddress,vol:BlockDeviceMappings}'
aws ec2 describe-security-groups --group-ids <sg-id>
aws ec2 describe-volumes --volume-ids <vol-id>
aws ec2 describe-addresses            # Elastic IP?
aws route53 list-hosted-zones         # domain?
```
On the host: `du -sh data/chroma`, `docker compose ps`, `git rev-parse HEAD`.
Record instance type, disk size, exposed ports and `data/chroma` size.

## 2. Baseline capture (current system, before any change)

```bash
python scripts/test_orchestration.py --output baseline.json
python scripts/test_orchestration.py --minimal --output baseline-minimal.json
python scripts/test_rag.py --health-check > baseline-rag.txt     # expect 13,349 docs total
```

## 3. Provision (company account)

### 3.1 S3 bucket for the Chroma copy
Private bucket, SSE-S3 or SSE-KMS, block public access, lifecycle rule to expire the archive after sign-off.

### 3.2 IAM role + instance profile
Trust policy: `ec2.amazonaws.com`. Permissions (replace placeholders):

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Sid": "InvokeOpus",
      "Effect": "Allow",
      "Action": ["bedrock:InvokeModel", "bedrock:InvokeModelWithResponseStream"],
      "Resource": [
        "arn:aws:bedrock:<region>:<account-id>:inference-profile/<BEDROCK_MODEL_ID>",
        "arn:aws:bedrock:*::foundation-model/<opus-5.5-foundation-model-id>"
      ]
    },
    {
      "Sid": "HealthCheck",
      "Effect": "Allow",
      "Action": ["bedrock:GetFoundationModel", "bedrock:GetInferenceProfile"],
      "Resource": "*"
    },
    {
      "Sid": "ChromaMigrationRead",
      "Effect": "Allow",
      "Action": ["s3:GetObject", "s3:ListBucket"],
      "Resource": ["arn:aws:s3:::<bucket>", "arn:aws:s3:::<bucket>/*"]
    }
  ]
}
```
Notes:
- `Converse` is authorised by `bedrock:InvokeModel`.
- With a cross-region inference profile, the foundation-model ARN must be allowed in every
  region the profile routes to (hence `*` region above). With a plain foundation-model ID,
  drop the inference-profile ARN and pin the region.
- `/api/health` calls `GetInferenceProfile` (IDs with a geo prefix such as `us.`/`eu.`/`global.`,
  or an `inference-profile/` ARN) or `GetFoundationModel` (plain model IDs). It never invokes the model.

### 3.3 Bedrock model access
Bedrock console → Model access → enable Claude Opus 5.5 in `AWS_REGION` (and the source regions
of the inference profile). Check quotas in Service Quotas → Amazon Bedrock.

### 3.4 Security group
- Inbound `3000/tcp` from the allowed CIDR only (or from the ALB security group if TLS is required).
- Prefer SSM Session Manager over opening `22`.
- Do **not** expose `8001` (Chroma), `8002` (API) or `8004` (MCP) — they have no auth.

### 3.5 EC2 instance
- CPU-only, e.g. `m7i.xlarge` (4 vCPU, 16 GB): embeddings run on CPU and the LLM is remote.
- Ubuntu 22.04/24.04; root volume ≥ 40 GB.
- Separate gp3 EBS volume for `data/chroma`, mounted at `<repo>/data/chroma` (size from §1 + headroom).
- Attach the instance profile from §3.2.
- **IMDSv2 hop limit = 2**, so containers on the Docker bridge network can obtain the
  instance-role credentials:
  ```bash
  aws ec2 modify-instance-metadata-options --instance-id <id> \
    --http-tokens required --http-put-response-hop-limit 2
  ```
- Install Docker + Compose plugin (README "Fresh Server Setup Guide").
- Optional: ALB + ACM certificate in front of port 3000 if TLS is required.

## 4. Copy ChromaDB data via S3

On the **old** host:
```bash
docker compose stop vector-db
tar -czf chroma.tgz -C data chroma
sha256sum chroma.tgz
aws s3 cp chroma.tgz s3://<bucket>/chroma.tgz   # cross-account: bucket policy granting the personal account PutObject, or a presigned URL
docker compose start vector-db
```
On the **new** host:
```bash
aws s3 cp s3://<bucket>/chroma.tgz .
sha256sum chroma.tgz                              # must match
tar -xzf chroma.tgz -C data
```

## 5. Deploy

```bash
git clone git@github.com:gharaehs/ai-health-orchestration.git && cd ai-health-orchestration
cat > .env <<'ENV'
LLM_PROVIDER=bedrock
AWS_REGION=<region>
BEDROCK_MODEL_ID=<model-or-inference-profile-id>
ENV
make start
make test-llm                                     # one-shot call through the shared client
curl -s http://localhost:8002/api/health | python3 -m json.tool
```
`.env` is git-ignored and read by Docker Compose; it holds no secrets (credentials come from the instance role).

## 6. Validate

1. `/api/health` → `"status": "ok"`, `services.llm` = `{provider: bedrock, model: <id>, status: ok}`.
2. `python scripts/test_orchestration.py` and `--minimal` exit 0 with 4/4 agents succeeding.
3. Compare to the baseline: schemas validate, meal plan has 7 days, training session count equals
   the requested `training_days_per_week`.
4. `python scripts/test_rag.py --health-check` → 13,349 documents, same per-collection counts.
5. `python scripts/test_rag.py --evaluate` completes.
6. UI smoke test: Chat & Compare, Orchestrate, Knowledge Base; sidebar status dots green.
7. Unit tests: `pip install -r requirements-dev.txt && make test`.

## 7. Cut over and decommission

1. Point users/DNS at the new host.
2. Keep the personal-account stack **stopped but intact** until sign-off.
3. After sign-off: terminate the old instance, delete its EBS volumes and Elastic IP, and delete
   the S3 archive.

## 8. Rollback

- Before decommission: start the old stack (`make start` on the old host) and point users back.
- The vLLM path still exists for a GPU host: `make start-vllm` (Llama weights must be placed in
  `models/llama` manually — `scripts/download_model.sh` was removed).
- Code rollback: redeploy the previous git revision.

## 9. Observability (CloudWatch namespace `AWS/Bedrock`, dimension `ModelId`)

| Metric | Alarm suggestion |
|--------|------------------|
| `Invocations` | — (usage trend) |
| `InvocationLatency` | p90 > 60 s |
| `InvocationThrottles` | > 0 for 5 min |
| `InvocationClientErrors` / `InvocationServerErrors` | > 0 for 5 min |
| `InputTokenCount` / `OutputTokenCount` | budget-based |

API logs (`docker compose logs api`) include per-call latency and input/output token counts.
Enable Bedrock model-invocation logging only if permitted by the data classification —
prompts contain health data.

## 10. Cost estimate

Calls per orchestration run: Lab 1–2, Nutrition 2, Training 1–2, Grocery up to 9 → **~13–15**.
Each `/api/chat` request → **2** calls (base + RAG). `/api/health` makes no model call.

Approximate tokens per run (prompt sizes in README, ~4 chars/token):
- input ≈ 13–15 calls × ~2–5k tokens ≈ **40–60k input tokens**
- output ≤ 2,048 per call, typically **15–25k output tokens**

```
cost/run ≈ input_tokens/1e6 × <Opus 5.5 input $/MTok> + output_tokens/1e6 × <Opus 5.5 output $/MTok>
```
Fill in current Bedrock pricing for the chosen region, add EC2, EBS and data transfer, and verify
against `InputTokenCount`/`OutputTokenCount` after a few runs.

## Known limitations (unchanged, out of scope)

- Single host, in-memory job store, one pipeline at a time — jobs are lost on restart.
- Frontend is served by the Vite dev server; API CORS allows all origins.
- Nutrition/Grocery 2,048-token workarounds remain (follow-up simplification).
