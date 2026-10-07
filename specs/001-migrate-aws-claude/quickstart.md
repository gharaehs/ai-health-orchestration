# Quickstart & Validation: Company AWS + Claude Opus 5.5

These validation scenarios prove the migration works. The interfaces are described in
[contracts/](./contracts/) and [data-model.md](./data-model.md).

## 1. Prerequisites (company AWS account)

1. **Model access**: Claude Opus 5.5 is enabled in Amazon Bedrock for the company region.
   Find the inference profile ID:
   ```bash
   aws bedrock list-inference-profiles --region "$AWS_REGION" \
     --query "inferenceProfileSummaries[?contains(inferenceProfileName, 'Opus')].[inferenceProfileId,status]" --output table
   ```
2. **IAM role** attached to the instance, with `bedrock:InvokeModel` on that profile ARN and its
   foundation-model ARNs.
3. **EC2 instance**: CPU-only (for example `m7i.xlarge`), Ubuntu 22.04, at least 40 GB gp3,
   Docker and the Compose plugin. The security group exposes the same ports as today except 8000.
4. **IMDS hop limit set to 2** so containers can use the role:
   ```bash
   aws ec2 modify-instance-metadata-options --instance-id <id> \
     --http-tokens required --http-put-response-hop-limit 2
   ```

## 2. Migrate data (not in git)

```bash
# on the personal instance
tar czf chroma.tgz data/chroma && tar czf corpus.tgz data/corpus
# copy to the company instance (scp/rsync, or via a temporary S3 bucket in the company account)
# on the company instance, inside the repo clone
tar xzf chroma.tgz && tar xzf corpus.tgz
```

Before copying, record the source counts:
`curl -s localhost:8001/api/v2/tenants/default_tenant/databases/default_database/collections`
plus the count for each collection.

## 3. Configure and start

```bash
cat > .env <<EOF
AWS_REGION=<company-region>
BEDROCK_MODEL_ID=<inference-profile-id-from-step-1>
EOF
docker compose up -d --build
docker compose ps      # expect: vector-db, mcp-rag, api, frontend (no llm service)
```

## 4. Validation scenarios

| # | Scenario | Command / action | Expected | Covers |
|---|----------|------------------|----------|--------|
| V1 | Credentials and model reachable | `curl -s 'localhost:8002/api/health?deep=true'` | `status: ok`; `services.llm.provider == "bedrock"`; `services.chromadb.collections == 4` | FR-002, FR-009, FR-012 |
| V2 | No secrets or GPU | `grep -rE 'AKIA\|aws_secret' . --exclude-dir=node_modules`; `docker compose config \| grep -i nvidia` | No matches | FR-009, FR-011 |
| V3 | Knowledge base intact | Per-collection counts from the collections endpoint | Equal to the source; 13,349 total | SC-005, FR-006 |
| V4 | Retrieval equivalence | Run 10 reference queries on `POST /api/search` against old and new deployments | Identical top-5 IDs per query | SC-006 |
| V5 | End-to-end orchestration | `python scripts/test_orchestration.py` (reference profile), 5 runs | All 4 agents succeed; each output validates against `agents/schemas.py` | SC-001, FR-003, FR-004 |
| V6 | Constraints respected | Review V5 outputs: lab `dietary_constraints` and `training_constraints` reflected in meals and sessions | 100% respected | SC-007 |
| V7 | Duration | `duration_s` from `/api/orchestrate/status/{id}` in V5 | ≤ 420 s | SC-003 |
| V8 | Reliability | 20 consecutive orchestrations over varied profiles | ≥ 19 complete | SC-002 |
| V9 | Grounding active | `docker compose logs api \| grep -E "Retrieved [0-9]+ chunks via MCP"` during V5 | Present for Lab, Nutrition and Training agents | FR-005 |
| V10 | Retrieval failure is visible | `docker compose stop mcp-rag`; start an orchestration | Job `failed`, error names `RetrievalUnavailableError`; then `docker compose start mcp-rag` | Edge case, R11 |
| V11 | Model failure is bounded | Run with an invalid `BEDROCK_MODEL_ID` | Job `failed` within the retry or timeout bound with `LLMUnavailableError`; no hang | FR-010 |
| V12 | Dashboard | Open `http://<ip>:3000`; use all 6 views; submit a Chat & Compare question | All views load; both answers come back; labels show Claude Opus 5.5 / Bedrock | SC-004, FR-008 |
| V13 | Unit tests | `pytest tests/` | Shared client and agent tests pass with Bedrock stubbed | Principle V |

## 5. Decommission the personal account (after V1–V12 pass)

1. Confirm that V3 passed on the company instance, so the data copy is complete.
2. In the personal account, terminate the EC2 instance, then delete its EBS volumes and
   snapshots, any Elastic IP, and the security groups.
3. Check that Cost Explorer shows no further charges for these resources (SC-008).
