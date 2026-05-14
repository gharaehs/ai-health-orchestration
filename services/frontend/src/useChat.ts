import { useState } from 'react'
import { ChatResult, HealthProfile } from './types'

const QUERY_TYPES: Record<string, string> = {
  meal_plan:    'meal_plan',
  gym_program:  'gym_program',
  lab_analysis: 'lab_analysis',
  general:      'general',
}

function detectQueryType(query: string): string {
  const q = query.toLowerCase()
  if (q.match(/meal|food|eat|protein|carb|calorie|recipe|diet/)) return 'meal_plan'
  if (q.match(/workout|gym|exercise|lift|train|rep|set|program/)) return 'gym_program'
  if (q.match(/blood|lab|ldl|hdl|glucose|creatinine|marker|test/)) return 'lab_analysis'
  return 'general'
}

export function useChat() {
  const [result, setResult] = useState<ChatResult | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  async function sendQuery(query: string, profile: HealthProfile, topK = 6) {
    setLoading(true)
    setError(null)
    setResult(null)

    try {
      const res = await fetch('/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          query,
          health_profile: profile,
          query_type: detectQueryType(query),
          top_k: topK,
        }),
      })
      if (!res.ok) throw new Error(`API error: ${res.status}`)
      const data: ChatResult = await res.json()
      setResult(data)
    } catch (e: any) {
      setError(e.message)
    } finally {
      setLoading(false)
    }
  }

  return { result, loading, error, sendQuery }
}
