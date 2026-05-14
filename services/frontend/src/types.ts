export interface Source {
  collection: string
  excerpt: string
  score: number
  metadata: Record<string, string>
}

export interface LLMResponse {
  content: string
  latency_s: number
  prompt_tokens: number
  completion_tokens: number
  error: string | null
}

export interface ChatResult {
  base_response: LLMResponse
  rag_response: LLMResponse
  retrieval: {
    chunks_retrieved: number
    collections_queried: string[]
    retrieval_latency_s: number
    sources: Source[]
  }
  metrics: {
    grounding_score: number
    base_score: number
    rag_improvement: number
    latency_delta_s: number
  }
}

export interface HealthProfile {
  weight_kg: number | ''
  height_cm: number | ''
  body_fat_pct: number | ''
  lbm_kg: number | ''
  ldl_mmol: number | ''
  hdl_mmol: number | ''
  glucose_mmol: number | ''
  creatinine: number | ''
  goal: string
  notes: string
}
