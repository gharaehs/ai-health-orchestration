import { useState } from 'react'
import { useChat } from './useChat'
import { HealthProfile, Source } from './types'
import {
  MessageSquare, BarChart2, Database, User, Settings,
  Send, ChevronDown, ChevronUp, Zap, BookOpen, Activity
} from 'lucide-react'
import {
  LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer, BarChart, Bar
} from 'recharts'

const DEFAULT_PROFILE: HealthProfile = {
  weight_kg: '', height_cm: '', body_fat_pct: '', lbm_kg: '',
  ldl_mmol: '', hdl_mmol: '', glucose_mmol: '', creatinine: '',
  goal: 'muscle_gain', notes: '',
}

type View = 'chat' | 'analytics' | 'kb' | 'profile' | 'config' | 'orchestrate'

// ── Orchestration types ───────────────────────────────────────────────────────

interface AgentResult { agent: string; status: string; duration_s: number; error?: string }

interface OrchestrationResult {
  lab_analysis?: {
    estimated_tdee?: number
    recommended_calories?: number
    dietary_constraints: string[]
    training_constraints: string[]
    overall_health_summary: string
    analysed_markers: Array<{ name: string; value: number; unit: string; status: string; reference_range: string; interpretation: string }>
    sources_used: string[]
  }
  nutrition?: {
    avg_daily_calories: number
    avg_daily_protein_g: number
    avg_daily_carbs_g: number
    avg_daily_fat_g: number
    weekly_plan: Array<{
      day: string
      total_calories: number
      total_protein_g: number
      total_carbs_g: number
      total_fat_g: number
      meals: Array<{ name: string; recipe_name: string; calories_kcal: number; protein_g: number; carbs_g: number; fat_g: number; ingredients: string[]; instructions: string }>
    }>
  }
  training?: {
    rest_days: string[]
    progression_scheme: { principle: string; week_2_adjustment: string; week_3_adjustment: string; week_4_adjustment: string }
    weekly_program: Array<{ day: string; session_type: string; duration_minutes: number; exercises: Array<{ name: string; sets: number; reps: string; rest_seconds: number; notes?: string }> }>
  }
  grocery?: {
    total_items: number
    estimated_weekly_cost_eur?: number
    shopping_notes: string[]
    items_by_category: Record<string, Array<{ name: string; total_quantity: string; estimated_cost_eur?: number }>>
  }
  agent_results: AgentResult[]
  total_duration_s: number
}

// ── Orchestration request builder ─────────────────────────────────────────────

function buildOrchestrationRequest(profile: HealthProfile) {
  const bloodMarkers = []
  if (profile.ldl_mmol)    bloodMarkers.push({ name: 'LDL Cholesterol', value: +profile.ldl_mmol, unit: 'mmol/L' })
  if (profile.hdl_mmol)    bloodMarkers.push({ name: 'HDL Cholesterol', value: +profile.hdl_mmol, unit: 'mmol/L' })
  if (profile.glucose_mmol) bloodMarkers.push({ name: 'Fasting Glucose', value: +profile.glucose_mmol, unit: 'mmol/L' })

  const goalMap: Record<string, string> = {
    muscle_gain: 'muscle_gain', fat_loss: 'fat_loss',
    maintenance: 'maintenance', performance: 'performance',
  }

  return {
    profile: {
      age: 38, sex: 'male',
      height_cm: profile.height_cm ? +profile.height_cm : 178,
      scale_metrics: {
        weight_kg: profile.weight_kg ? +profile.weight_kg : 80,
        body_fat_pct: profile.body_fat_pct ? +profile.body_fat_pct : null,
        muscle_mass_kg: profile.lbm_kg ? +profile.lbm_kg : null,
        bmi: null,
      },
      blood_markers: bloodMarkers,
      medical_history: profile.notes ? [profile.notes] : [],
      medications: [],
      allergies: [],
    },
    goals: {
      primary_goal: goalMap[profile.goal] || 'maintenance',
      activity_level: 'moderately_active',
      fitness_level: 'intermediate',
      training_days_per_week: 4,
      dietary_preferences: [],
      notes: null,
    },
  }
}

// ── Main App ──────────────────────────────────────────────────────────────────

export default function App() {
  const [view, setView] = useState<View>('chat')
  const [query, setQuery] = useState('')
  const [profile, setProfile] = useState<HealthProfile>(DEFAULT_PROFILE)
  const [topK, setTopK] = useState(6)
  const [expandedSource, setExpandedSource] = useState<number | null>(null)
  const [history, setHistory] = useState<Array<{ query: string; grounding: number; baseLatency: number; ragLatency: number }>>([])
  const [kbQuery, setKbQuery] = useState('')
  const [kbResults, setKbResults] = useState<Source[]>([])
  const [kbLoading, setKbLoading] = useState(false)

  // Orchestration state
  const [orchLoading, setOrchLoading] = useState(false)
  const [orchResult, setOrchResult] = useState<OrchestrationResult | null>(null)
  const [orchError, setOrchError] = useState<string | null>(null)
  const [expandedAgent, setExpandedAgent] = useState<string | null>(null)

  const { result, loading, error, sendQuery } = useChat()

  async function handleSend() {
    if (!query.trim() || loading) return
    await sendQuery(query, profile, topK)
  }

  if (result && history[history.length - 1]?.query !== query) {
    setHistory(h => [...h, {
      query,
      grounding: result.metrics.grounding_score,
      baseLatency: result.base_response.latency_s,
      ragLatency: result.rag_response.latency_s,
    }])
  }

  async function handleKbSearch() {
    if (!kbQuery.trim()) return
    setKbLoading(true)
    try {
      const res = await fetch('/api/search', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query: kbQuery, top_k: 8 }),
      })
      const data = await res.json()
      setKbResults(data.sources || [])
    } catch {
      setKbResults([])
    } finally {
      setKbLoading(false)
    }
  }

  async function handleOrchestrate() {
    setOrchLoading(true)
    setOrchResult(null)
    setOrchError(null)
    setExpandedAgent(null)
    try {
      const body = buildOrchestrationRequest(profile)

      // Step 1: Start the job — returns immediately with job_id
      const startRes = await fetch('/api/orchestrate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
      })
      if (!startRes.ok) throw new Error(`Failed to start: HTTP ${startRes.status}`)
      const { job_id } = await startRes.json()

      // Step 2: Poll status every 10 seconds
      while (true) {
        await new Promise(r => setTimeout(r, 10000))
        const statusRes = await fetch(`/api/orchestrate/status/${job_id}`)
        if (!statusRes.ok) throw new Error(`Status check failed: HTTP ${statusRes.status}`)
        const statusData = await statusRes.json()

        if (statusData.status === 'complete') {
          // Step 3: Fetch result
          const resultRes = await fetch(`/api/orchestrate/result/${job_id}`)
          if (!resultRes.ok) throw new Error(`Result fetch failed: HTTP ${resultRes.status}`)
          const data: OrchestrationResult = await resultRes.json()
          setOrchResult(data)
          const first = data.agent_results.find(r => r.status === 'success')
          if (first) setExpandedAgent(first.agent)
          break
        } else if (statusData.status === 'failed') {
          throw new Error(statusData.error || 'Pipeline failed')
        }
        // status is 'pending' or 'running' — keep polling
      }
    } catch (e: any) {
      setOrchError(e.message)
    } finally {
      setOrchLoading(false)
    }
  }

  const navItems: Array<{ id: View; icon: any; label: string }> = [
    { id: 'chat',        icon: MessageSquare, label: 'Chat & Compare' },
    { id: 'orchestrate', icon: Activity,      label: 'Orchestrate' },
    { id: 'analytics',   icon: BarChart2,     label: 'Analytics' },
    { id: 'kb',          icon: Database,      label: 'Knowledge Base' },
    { id: 'profile',     icon: User,          label: 'Health Profile' },
    { id: 'config',      icon: Settings,      label: 'Config' },
  ]

  return (
      <div className="flex h-screen bg-gray-950 text-gray-100 font-sans text-sm">

        {/* Sidebar */}
        <div className="w-52 flex-shrink-0 bg-gray-900 border-r border-gray-800 flex flex-col">
          <div className="p-4 border-b border-gray-800">
            <div className="font-semibold text-white">Health AI</div>
            <div className="text-xs text-gray-500 mt-0.5">Orchestration System</div>
          </div>
          <nav className="flex-1 py-2">
            {navItems.map(({ id, icon: Icon, label }) => (
                <button
                    key={id}
                    onClick={() => setView(id)}
                    className={`w-full flex items-center gap-2.5 px-4 py-2.5 text-left transition-colors ${
                        view === id
                            ? 'bg-gray-800 text-white border-l-2 border-blue-500 pl-3.5'
                            : 'text-gray-400 hover:bg-gray-800 hover:text-gray-200'
                    }`}
                >
                  <Icon size={15} />
                  <span className="text-xs">{label}</span>
                </button>
            ))}
          </nav>
          <div className="p-3 border-t border-gray-800 space-y-1.5">
            <StatusDot label="vLLM :8000" />
            <StatusDot label="ChromaDB :8001" />
            <StatusDot label="API :8002" />
          </div>
        </div>

        {/* Main */}
        <div className="flex-1 flex flex-col overflow-hidden">
          <div className="h-11 flex-shrink-0 bg-gray-900 border-b border-gray-800 flex items-center justify-between px-5">
          <span className="font-medium text-white">
            {navItems.find(n => n.id === view)?.label}
          </span>
            <div className="flex items-center gap-2">
              <span className="text-xs px-2 py-0.5 rounded-full bg-blue-900 text-blue-300">Llama 3.1 8B</span>
              <span className="text-xs px-2 py-0.5 rounded-full bg-purple-900 text-purple-300">LoRA: health-v1</span>
            </div>
          </div>

          <div className="flex-1 overflow-y-auto p-5">

            {/* ── CHAT VIEW ── */}
            {view === 'chat' && (
                <div className="flex flex-col gap-4 h-full">
                  {result && (
                      <div className="grid grid-cols-4 gap-3">
                        <MetricCard label="Base latency"  value={`${result.base_response.latency_s}s`} />
                        <MetricCard label="RAG latency"   value={`${result.rag_response.latency_s}s`} />
                        <MetricCard label="Chunks retrieved" value={String(result.retrieval.chunks_retrieved)} />
                        <MetricCard label="Grounding score"
                                    value={`${(result.metrics.grounding_score * 100).toFixed(0)}%`}
                                    highlight={result.metrics.grounding_score > 0.5} />
                      </div>
                  )}
                  <div className="bg-gray-900 border border-gray-800 rounded-lg p-4">
                    <div className="flex gap-2 items-start">
                  <textarea
                      className="flex-1 bg-gray-800 border border-gray-700 rounded-md px-3 py-2 text-sm resize-none focus:outline-none focus:border-blue-500 text-gray-100 placeholder-gray-500"
                      rows={3}
                      placeholder="Ask about your health data, meal plan, gym program, or lab results…"
                      value={query}
                      onChange={e => setQuery(e.target.value)}
                      onKeyDown={e => { if (e.key === 'Enter' && e.metaKey) handleSend() }}
                  />
                      <button
                          onClick={handleSend}
                          disabled={loading || !query.trim()}
                          className="px-4 py-2 bg-blue-600 hover:bg-blue-500 disabled:bg-gray-700 disabled:text-gray-500 rounded-md text-white transition-colors flex items-center gap-1.5"
                      >
                        <Send size={14} />
                        {loading ? 'Running…' : 'Send'}
                      </button>
                    </div>
                    <div className="flex items-center gap-3 mt-2 text-xs text-gray-500">
                  <span>top-k: <input type="number" value={topK} onChange={e => setTopK(+e.target.value)}
                                      className="w-10 bg-gray-800 border border-gray-700 rounded px-1 text-gray-300" /></span>
                      <span>Goal: <span className="text-gray-300">{profile.goal}</span></span>
                      <span className="ml-auto text-gray-600">⌘+Enter to send</span>
                    </div>
                  </div>
                  {error && <div className="bg-red-950 border border-red-800 rounded-lg p-3 text-red-300 text-xs">{error}</div>}
                  {loading && (
                      <div className="bg-gray-900 border border-gray-800 rounded-lg p-6 text-center text-gray-500">
                        <div className="animate-pulse">Running parallel RAG + base calls…</div>
                        <div className="text-xs mt-1 text-gray-600">This takes 20–60s on T4</div>
                      </div>
                  )}
                  {result && (
                      <>
                        <div className="grid grid-cols-2 gap-4">
                          <ResponsePanel title="Base model (no RAG)" content={result.base_response.content}
                                         tokens={result.base_response.prompt_tokens} latency={result.base_response.latency_s} variant="base" />
                          <ResponsePanel title="RAG-augmented response" content={result.rag_response.content}
                                         tokens={result.rag_response.prompt_tokens} latency={result.rag_response.latency_s} variant="rag" />
                        </div>
                        <div className="bg-gray-900 border border-gray-800 rounded-lg overflow-hidden">
                          <div className="px-4 py-2.5 border-b border-gray-800 flex items-center justify-between">
                      <span className="text-xs font-medium text-gray-400 uppercase tracking-wide flex items-center gap-1.5">
                        <BookOpen size={13} /> Retrieved sources ({result.retrieval.chunks_retrieved})
                      </span>
                            <span className="text-xs text-gray-600">retrieval: {result.retrieval.retrieval_latency_s}s</span>
                          </div>
                          <div className="divide-y divide-gray-800 max-h-96 overflow-y-auto">
                            {result.retrieval.sources.map((src, i) => (
                                <div key={i} className="p-3">
                                  <div className="flex items-start gap-3 cursor-pointer"
                                       onClick={() => setExpandedSource(expandedSource === i ? null : i)}>
                            <span className={`text-xs font-mono font-semibold min-w-[42px] ${
                                src.score > 0.7 ? 'text-green-400' : src.score > 0.5 ? 'text-yellow-400' : 'text-gray-500'
                            }`}>{src.score.toFixed(3)}</span>
                                    <div className="flex-1 min-w-0">
                                      <div className="flex items-center gap-2">
                                        <span className="text-xs px-1.5 py-0.5 rounded bg-gray-800 text-gray-400">{src.collection}</span>
                                        {src.metadata?.source && <span className="text-xs text-gray-500 truncate">{src.metadata.source}</span>}
                                      </div>
                                      <p className="text-xs text-gray-400 mt-1 line-clamp-2">{src.excerpt}</p>
                                    </div>
                                    {expandedSource === i ? <ChevronUp size={13} className="text-gray-600 flex-shrink-0" /> : <ChevronDown size={13} className="text-gray-600 flex-shrink-0" />}
                                  </div>
                                  {expandedSource === i && (
                                      <div className="mt-2 ml-12 p-2 bg-gray-800 rounded text-xs text-gray-300 leading-relaxed">
                                        {src.excerpt}
                                      </div>
                                  )}
                                </div>
                            ))}
                          </div>
                        </div>
                        <div className="bg-gray-900 border border-gray-800 rounded-lg p-4">
                          <div className="text-xs font-medium text-gray-400 uppercase tracking-wide mb-3">Quality metrics</div>
                          <div className="grid grid-cols-4 gap-4">
                            <ScoreBox label="Grounding" value={result.metrics.grounding_score} color="green" />
                            <ScoreBox label="Base score" value={result.metrics.base_score} color="gray" />
                            <ScoreBox label="RAG improvement" value={result.metrics.rag_improvement} color="blue" />
                            <div className="text-center">
                              <div className={`text-xl font-semibold ${result.metrics.latency_delta_s > 0 ? 'text-yellow-400' : 'text-green-400'}`}>
                                {result.metrics.latency_delta_s > 0 ? '+' : ''}{result.metrics.latency_delta_s}s
                              </div>
                              <div className="text-xs text-gray-500 mt-1">Latency delta</div>
                            </div>
                          </div>
                        </div>
                      </>
                  )}
                </div>
            )}

            {/* ── ORCHESTRATE VIEW ── */}
            {view === 'orchestrate' && (
                <div className="space-y-4 max-w-4xl">

                  {/* Run button + status */}
                  <div className="bg-gray-900 border border-gray-800 rounded-lg p-4">
                    <div className="flex items-start justify-between gap-4">
                      <div>
                        <div className="text-sm font-medium text-white mb-1">Multi-Agent Pipeline</div>
                        <div className="text-xs text-gray-500">
                          Runs 4 agents sequentially: Lab Analysis → Nutrition → Training → Grocery.
                          Uses health profile from the Profile tab. Takes ~7 min on T4.
                        </div>
                      </div>
                      <button
                          onClick={handleOrchestrate}
                          disabled={orchLoading}
                          className="flex-shrink-0 px-5 py-2 bg-blue-600 hover:bg-blue-500 disabled:bg-gray-700 disabled:text-gray-500 rounded-md text-white text-sm font-medium transition-colors flex items-center gap-2"
                      >
                        <Activity size={14} />
                        {orchLoading ? 'Running pipeline…' : 'Run Pipeline'}
                      </button>
                    </div>

                    {/* Agent progress */}
                    {(orchLoading || orchResult) && (
                        <div className="mt-4 grid grid-cols-4 gap-2">
                          {['LabAnalysisAgent', 'NutritionAgent', 'TrainingAgent', 'GroceryAgent'].map(name => {
                            const agentResult = orchResult?.agent_results.find(r => r.agent === name)
                            const label = name.replace('Agent', '')
                            if (!agentResult && orchLoading) {
                              return (
                                  <div key={name} className="bg-gray-800 rounded-lg p-3 text-center">
                                    <div className="text-yellow-400 text-lg mb-1 animate-pulse">⟳</div>
                                    <div className="text-xs text-gray-400">{label}</div>
                                    <div className="text-xs text-gray-600 mt-0.5">waiting…</div>
                                  </div>
                              )
                            }
                            if (!agentResult) return null
                            const icon = agentResult.status === 'success' ? '✅' : agentResult.status === 'failed' ? '❌' : '⏭️'
                            return (
                                <div key={name} className="bg-gray-800 rounded-lg p-3 text-center">
                                  <div className="text-lg mb-1">{icon}</div>
                                  <div className="text-xs text-gray-300 font-medium">{label}</div>
                                  <div className="text-xs text-gray-500 mt-0.5">{agentResult.duration_s.toFixed(1)}s</div>
                                </div>
                            )
                          })}
                        </div>
                    )}

                    {orchResult && (
                        <div className="mt-3 flex items-center gap-3 text-xs text-gray-500">
                          <span>Total: <span className="text-white font-medium">{orchResult.total_duration_s.toFixed(0)}s</span></span>
                          <span>·</span>
                          <span>{orchResult.agent_results.filter(r => r.status === 'success').length}/{orchResult.agent_results.length} agents succeeded</span>
                        </div>
                    )}
                  </div>

                  {orchError && (
                      <div className="bg-red-950 border border-red-800 rounded-lg p-3 text-red-300 text-xs">{orchError}</div>
                  )}

                  {orchLoading && (
                      <div className="bg-gray-900 border border-gray-800 rounded-lg p-8 text-center text-gray-500">
                        <div className="animate-pulse text-base mb-2">Pipeline running…</div>
                        <div className="text-xs text-gray-600">Each agent calls vLLM with RAG-augmented prompts</div>
                        <div className="text-xs text-gray-700 mt-1">~7 minutes on T4 · check server logs for progress</div>
                      </div>
                  )}

                  {/* ── Agent output panels ── */}
                  {orchResult && (
                      <div className="space-y-3">

                        {/* Lab Analysis */}
                        {orchResult.lab_analysis && (
                            <AgentPanel
                                title="Lab Analysis"
                                icon="🔬"
                                expanded={expandedAgent === 'LabAnalysisAgent'}
                                onToggle={() => setExpandedAgent(expandedAgent === 'LabAnalysisAgent' ? null : 'LabAnalysisAgent')}
                                summary={`TDEE ${orchResult.lab_analysis.estimated_tdee} kcal → Target ${orchResult.lab_analysis.recommended_calories} kcal · ${orchResult.lab_analysis.dietary_constraints.length} dietary constraints`}
                            >
                              <div className="space-y-3">
                                <p className="text-xs text-gray-300 leading-relaxed">{orchResult.lab_analysis.overall_health_summary}</p>
                                <div className="grid grid-cols-2 gap-3">
                                  <div>
                                    <div className="text-xs font-medium text-gray-400 mb-2">Dietary Constraints</div>
                                    <ul className="space-y-1">
                                      {orchResult.lab_analysis.dietary_constraints.map((c, i) => (
                                          <li key={i} className="text-xs text-gray-300 flex gap-1.5"><span className="text-orange-400 flex-shrink-0">•</span>{c}</li>
                                      ))}
                                    </ul>
                                  </div>
                                  <div>
                                    <div className="text-xs font-medium text-gray-400 mb-2">Training Constraints</div>
                                    <ul className="space-y-1">
                                      {orchResult.lab_analysis.training_constraints.map((c, i) => (
                                          <li key={i} className="text-xs text-gray-300 flex gap-1.5"><span className="text-blue-400 flex-shrink-0">•</span>{c}</li>
                                      ))}
                                    </ul>
                                  </div>
                                </div>
                                {orchResult.lab_analysis.analysed_markers.length > 0 && (
                                    <div>
                                      <div className="text-xs font-medium text-gray-400 mb-2">Blood Markers</div>
                                      <div className="space-y-1">
                                        {orchResult.lab_analysis.analysed_markers.map((m, i) => {
                                          const statusColor: Record<string, string> = {
                                            optimal: 'text-green-400', elevated: 'text-red-400',
                                            'borderline-high': 'text-yellow-400', low: 'text-orange-400',
                                            deficient: 'text-red-400', borderline: 'text-yellow-400',
                                          }
                                          return (
                                              <div key={i} className="flex items-start gap-3 text-xs bg-gray-800 rounded p-2">
                                                <span className="text-gray-400 w-32 flex-shrink-0">{m.name}</span>
                                                <span className="text-white font-mono">{m.value} {m.unit}</span>
                                                <span className={`${statusColor[m.status] || 'text-gray-400'} capitalize`}>{m.status}</span>
                                                <span className="text-gray-500 flex-1">{m.reference_range}</span>
                                              </div>
                                          )
                                        })}
                                      </div>
                                    </div>
                                )}
                                {orchResult.lab_analysis.sources_used.length > 0 && (
                                    <div className="text-xs text-gray-600">Sources: {orchResult.lab_analysis.sources_used.join(' · ')}</div>
                                )}
                              </div>
                            </AgentPanel>
                        )}

                        {/* Nutrition */}
                        {orchResult.nutrition && (
                            <AgentPanel
                                title="Nutrition Plan"
                                icon="🥗"
                                expanded={expandedAgent === 'NutritionAgent'}
                                onToggle={() => setExpandedAgent(expandedAgent === 'NutritionAgent' ? null : 'NutritionAgent')}
                                summary={`${orchResult.nutrition.avg_daily_calories} kcal/day · P:${orchResult.nutrition.avg_daily_protein_g}g · C:${orchResult.nutrition.avg_daily_carbs_g}g · F:${orchResult.nutrition.avg_daily_fat_g}g`}
                            >
                              <div className="space-y-2">
                                {orchResult.nutrition.weekly_plan.map((day, i) => (
                                    <div key={i} className="bg-gray-800 rounded-lg p-3">
                                      <div className="flex items-center justify-between mb-2">
                                        <span className="text-xs font-medium text-white">{day.day}</span>
                                        <span className="text-xs text-gray-500">{day.total_calories} kcal · P:{day.total_protein_g}g · C:{day.total_carbs_g}g · F:{day.total_fat_g}g</span>
                                      </div>
                                      <div className="space-y-1">
                                        {day.meals.map((meal, j) => (
                                            <div key={j} className="flex items-baseline gap-2 text-xs">
                                              <span className="text-gray-500 w-16 flex-shrink-0">{meal.name}</span>
                                              <span className="text-gray-300">{meal.recipe_name}</span>
                                              <span className="text-gray-600 ml-auto">{meal.calories_kcal} kcal</span>
                                            </div>
                                        ))}
                                      </div>
                                    </div>
                                ))}
                              </div>
                            </AgentPanel>
                        )}

                        {/* Training */}
                        {orchResult.training && (
                            <AgentPanel
                                title="Training Program"
                                icon="🏋️"
                                expanded={expandedAgent === 'TrainingAgent'}
                                onToggle={() => setExpandedAgent(expandedAgent === 'TrainingAgent' ? null : 'TrainingAgent')}
                                summary={`${orchResult.training.weekly_program.length} sessions/week · Rest: ${orchResult.training.rest_days.join(', ')} · ${orchResult.training.progression_scheme.principle}`}
                            >
                              <div className="space-y-2">
                                {orchResult.training.weekly_program.map((session, i) => (
                                    <div key={i} className="bg-gray-800 rounded-lg p-3">
                                      <div className="flex items-center justify-between mb-2">
                                        <span className="text-xs font-medium text-white">{session.day} — {session.session_type}</span>
                                        <span className="text-xs text-gray-500">{session.duration_minutes} min</span>
                                      </div>
                                      <div className="space-y-1">
                                        {session.exercises.map((ex, j) => (
                                            <div key={j} className="flex items-baseline gap-2 text-xs">
                                              <span className="text-gray-300 flex-1">{ex.name}</span>
                                              <span className="text-gray-500">{ex.sets}×{ex.reps}</span>
                                              <span className="text-gray-600">{ex.rest_seconds}s rest</span>
                                            </div>
                                        ))}
                                      </div>
                                    </div>
                                ))}
                                <div className="bg-gray-800 rounded-lg p-3 text-xs text-gray-400">
                                  <span className="font-medium text-gray-300">Progression: </span>
                                  {orchResult.training.progression_scheme.principle} · Week 4: {orchResult.training.progression_scheme.week_4_adjustment}
                                </div>
                              </div>
                            </AgentPanel>
                        )}

                        {/* Grocery */}
                        {orchResult.grocery && (
                            <AgentPanel
                                title="Grocery List"
                                icon="🛒"
                                expanded={expandedAgent === 'GroceryAgent'}
                                onToggle={() => setExpandedAgent(expandedAgent === 'GroceryAgent' ? null : 'GroceryAgent')}
                                summary={`${orchResult.grocery.total_items} items${orchResult.grocery.estimated_weekly_cost_eur ? ` · Est. €${orchResult.grocery.estimated_weekly_cost_eur}` : ''}`}
                            >
                              <div className="space-y-3">
                                {Object.entries(orchResult.grocery.items_by_category).filter(([, items]) => items.length > 0).map(([cat, items]) => (
                                    <div key={cat}>
                                      <div className="text-xs font-medium text-gray-400 mb-1">{cat}</div>
                                      <div className="grid grid-cols-2 gap-1">
                                        {items.map((item, i) => (
                                            <div key={i} className="flex justify-between text-xs bg-gray-800 rounded px-2 py-1">
                                              <span className="text-gray-300">{item.name}</span>
                                              <span className="text-gray-500">{item.total_quantity}</span>
                                            </div>
                                        ))}
                                      </div>
                                    </div>
                                ))}
                                {orchResult.grocery.shopping_notes.length > 0 && (
                                    <div>
                                      <div className="text-xs font-medium text-gray-400 mb-1">Shopping Notes</div>
                                      <ul className="space-y-1">
                                        {orchResult.grocery.shopping_notes.map((note, i) => (
                                            <li key={i} className="text-xs text-gray-400 flex gap-1.5"><span className="text-green-400">•</span>{note}</li>
                                        ))}
                                      </ul>
                                    </div>
                                )}
                              </div>
                            </AgentPanel>
                        )}
                      </div>
                  )}
                </div>
            )}

            {/* ── ANALYTICS VIEW ── */}
            {view === 'analytics' && (
                <div className="space-y-4">
                  <div className="bg-gray-900 border border-gray-800 rounded-lg p-4">
                    <div className="text-xs font-medium text-gray-400 uppercase tracking-wide mb-4">Latency over session</div>
                    {history.length === 0 ? (
                        <div className="text-gray-600 text-xs py-8 text-center">No requests yet — send a query from Chat view</div>
                    ) : (
                        <ResponsiveContainer width="100%" height={200}>
                          <LineChart data={history.map((h, i) => ({ i: i + 1, base: h.baseLatency, rag: h.ragLatency }))}>
                            <XAxis dataKey="i" stroke="#4b5563" tick={{ fontSize: 11 }} />
                            <YAxis stroke="#4b5563" tick={{ fontSize: 11 }} unit="s" />
                            <Tooltip contentStyle={{ background: '#111827', border: '1px solid #374151', fontSize: 12 }} />
                            <Line type="monotone" dataKey="base" stroke="#6b7280" name="Base" dot={false} />
                            <Line type="monotone" dataKey="rag"  stroke="#3b82f6" name="RAG"  dot={false} />
                          </LineChart>
                        </ResponsiveContainer>
                    )}
                  </div>
                  <div className="bg-gray-900 border border-gray-800 rounded-lg p-4">
                    <div className="text-xs font-medium text-gray-400 uppercase tracking-wide mb-4">Grounding score over session</div>
                    {history.length === 0 ? (
                        <div className="text-gray-600 text-xs py-8 text-center">No requests yet</div>
                    ) : (
                        <ResponsiveContainer width="100%" height={200}>
                          <BarChart data={history.map((h, i) => ({ i: i + 1, score: +(h.grounding * 100).toFixed(1) }))}>
                            <XAxis dataKey="i" stroke="#4b5563" tick={{ fontSize: 11 }} />
                            <YAxis stroke="#4b5563" tick={{ fontSize: 11 }} unit="%" />
                            <Tooltip contentStyle={{ background: '#111827', border: '1px solid #374151', fontSize: 12 }} />
                            <Bar dataKey="score" fill="#3b82f6" name="Grounding %" />
                          </BarChart>
                        </ResponsiveContainer>
                    )}
                  </div>
                </div>
            )}

            {/* ── KNOWLEDGE BASE VIEW ── */}
            {view === 'kb' && (
                <div className="space-y-4">
                  <div className="bg-gray-900 border border-gray-800 rounded-lg p-4">
                    <div className="text-xs font-medium text-gray-400 uppercase tracking-wide mb-3">Search knowledge base</div>
                    <div className="flex gap-2">
                      <input
                          className="flex-1 bg-gray-800 border border-gray-700 rounded-md px-3 py-2 text-sm focus:outline-none focus:border-blue-500 text-gray-100 placeholder-gray-500"
                          placeholder="Search across all collections…"
                          value={kbQuery}
                          onChange={e => setKbQuery(e.target.value)}
                          onKeyDown={e => { if (e.key === 'Enter') handleKbSearch() }}
                      />
                      <button onClick={handleKbSearch} disabled={kbLoading}
                              className="px-4 py-2 bg-blue-600 hover:bg-blue-500 disabled:bg-gray-700 rounded-md text-white text-sm transition-colors">
                        {kbLoading ? '…' : 'Search'}
                      </button>
                    </div>
                  </div>
                  {kbResults.length > 0 && (
                      <div className="bg-gray-900 border border-gray-800 rounded-lg divide-y divide-gray-800 max-h-[600px] overflow-y-auto">
                        {kbResults.map((src, i) => (
                            <div key={i} className="p-3">
                              <div className="flex items-start gap-3">
                        <span className={`text-xs font-mono font-semibold min-w-[42px] ${
                            src.score > 0.7 ? 'text-green-400' : src.score > 0.5 ? 'text-yellow-400' : 'text-gray-500'
                        }`}>{src.score.toFixed(3)}</span>
                                <div>
                                  <div className="flex items-center gap-2 mb-1">
                                    <span className="text-xs px-1.5 py-0.5 rounded bg-gray-800 text-gray-400">{src.collection}</span>
                                    {src.metadata?.source && <span className="text-xs text-gray-500">{src.metadata.source}</span>}
                                  </div>
                                  <p className="text-xs text-gray-300 leading-relaxed">{src.excerpt}</p>
                                </div>
                              </div>
                            </div>
                        ))}
                      </div>
                  )}
                </div>
            )}

            {/* ── HEALTH PROFILE VIEW ── */}
            {view === 'profile' && (
                <div className="max-w-2xl space-y-4">
                  <div className="bg-gray-900 border border-gray-800 rounded-lg p-4">
                    <div className="text-xs font-medium text-gray-400 uppercase tracking-wide mb-4">Body metrics</div>
                    <div className="grid grid-cols-2 gap-3">
                      {(['weight_kg','height_cm','body_fat_pct','lbm_kg'] as const).map(k => (
                          <ProfileField key={k} label={k.replace('_', ' ')} value={profile[k]}
                                        onChange={v => setProfile(p => ({ ...p, [k]: v }))} />
                      ))}
                    </div>
                  </div>
                  <div className="bg-gray-900 border border-gray-800 rounded-lg p-4">
                    <div className="text-xs font-medium text-gray-400 uppercase tracking-wide mb-4">Blood markers</div>
                    <div className="grid grid-cols-2 gap-3">
                      {(['ldl_mmol','hdl_mmol','glucose_mmol','creatinine'] as const).map(k => (
                          <ProfileField key={k} label={k.replace('_', ' ')} value={profile[k]}
                                        onChange={v => setProfile(p => ({ ...p, [k]: v }))} />
                      ))}
                    </div>
                  </div>
                  <div className="bg-gray-900 border border-gray-800 rounded-lg p-4">
                    <div className="text-xs font-medium text-gray-400 uppercase tracking-wide mb-3">Goal</div>
                    <div className="flex gap-2 flex-wrap">
                      {['muscle_gain','fat_loss','maintenance','performance'].map(g => (
                          <button key={g} onClick={() => setProfile(p => ({ ...p, goal: g }))}
                                  className={`px-3 py-1.5 rounded-md text-xs transition-colors ${
                                      profile.goal === g ? 'bg-blue-600 text-white' : 'bg-gray-800 text-gray-400 hover:bg-gray-700'
                                  }`}>{g.replace('_', ' ')}</button>
                      ))}
                    </div>
                  </div>
                  <div className="bg-gray-900 border border-gray-800 rounded-lg p-4">
                    <div className="text-xs font-medium text-gray-400 uppercase tracking-wide mb-3">Medical notes</div>
                    <textarea
                        className="w-full bg-gray-800 border border-gray-700 rounded-md px-3 py-2 text-sm resize-none focus:outline-none focus:border-blue-500 text-gray-100 placeholder-gray-500"
                        rows={3}
                        placeholder="Any relevant medical history, medications, or conditions…"
                        value={profile.notes}
                        onChange={e => setProfile(p => ({ ...p, notes: e.target.value }))}
                    />
                  </div>
                  <div className="bg-blue-950/30 border border-blue-800 rounded-lg p-3 text-xs text-blue-300">
                    💡 Profile values are used by the Orchestrate tab when running the full pipeline.
                  </div>
                </div>
            )}

            {/* ── CONFIG VIEW ── */}
            {view === 'config' && (
                <div className="max-w-lg space-y-4">
                  <div className="bg-gray-900 border border-gray-800 rounded-lg p-4 space-y-3">
                    <div className="text-xs font-medium text-gray-400 uppercase tracking-wide">RAG parameters</div>
                    <div className="flex items-center justify-between">
                      <span className="text-xs text-gray-400">Default top-k chunks</span>
                      <input type="number" value={topK} onChange={e => setTopK(+e.target.value)}
                             className="w-16 bg-gray-800 border border-gray-700 rounded px-2 py-1 text-sm text-gray-300 text-center" />
                    </div>
                  </div>
                  <div className="bg-gray-900 border border-gray-800 rounded-lg p-4 space-y-2">
                    <div className="text-xs font-medium text-gray-400 uppercase tracking-wide mb-3">System info</div>
                    <InfoRow label="Model" value="meta-llama/Meta-Llama-3.1-8B-Instruct" />
                    <InfoRow label="Quantization" value="BitsAndBytes 4-bit (NF4)" />
                    <InfoRow label="Context window" value="16,384 tokens" />
                    <InfoRow label="LoRA adapter" value="health-v1 (rank=16, alpha=32)" />
                    <InfoRow label="Embedding model" value="all-MiniLM-L6-v2 (CPU)" />
                    <InfoRow label="Vector DB" value="ChromaDB 1.5.8 — 13,349 docs" />
                    <InfoRow label="GPU" value="NVIDIA Tesla T4 — 16 GB VRAM" />
                  </div>
                </div>
            )}

          </div>
        </div>
      </div>
  )
}

// ── Sub-components ──

function AgentPanel({ title, icon, expanded, onToggle, summary, children }: {
  title: string; icon: string; expanded: boolean; onToggle: () => void;
  summary: string; children: React.ReactNode
}) {
  return (
      <div className="bg-gray-900 border border-gray-800 rounded-lg overflow-hidden">
        <button
            onClick={onToggle}
            className="w-full flex items-center justify-between px-4 py-3 hover:bg-gray-800 transition-colors"
        >
          <div className="flex items-center gap-2.5">
            <span>{icon}</span>
            <span className="text-sm font-medium text-white">{title}</span>
            <span className="text-xs text-gray-500">{summary}</span>
          </div>
          {expanded ? <ChevronUp size={14} className="text-gray-500 flex-shrink-0" /> : <ChevronDown size={14} className="text-gray-500 flex-shrink-0" />}
        </button>
        {expanded && (
            <div className="px-4 pb-4 pt-1 border-t border-gray-800">
              {children}
            </div>
        )}
      </div>
  )
}

function StatusDot({ label }: { label: string }) {
  return (
      <div className="flex items-center gap-1.5 text-xs text-gray-500">
        <span className="w-1.5 h-1.5 rounded-full bg-green-500 flex-shrink-0" />
        {label}
      </div>
  )
}

function MetricCard({ label, value, highlight = false }: { label: string; value: string; highlight?: boolean }) {
  return (
      <div className="bg-gray-900 border border-gray-800 rounded-lg p-3">
        <div className="text-xs text-gray-500 mb-1">{label}</div>
        <div className={`text-xl font-semibold ${highlight ? 'text-green-400' : 'text-white'}`}>{value}</div>
      </div>
  )
}

function ResponsePanel({ title, content, tokens, latency, variant }:
                       { title: string; content: string; tokens: number; latency: number; variant: 'base' | 'rag' }) {
  return (
      <div className={`bg-gray-900 border rounded-lg overflow-hidden ${variant === 'rag' ? 'border-blue-800' : 'border-gray-800'}`}>
        <div className={`px-4 py-2.5 border-b flex items-center justify-between ${variant === 'rag' ? 'border-blue-800 bg-blue-950/30' : 'border-gray-800'}`}>
        <span className="text-xs font-medium text-gray-300 flex items-center gap-1.5">
          {variant === 'rag' && <Zap size={12} className="text-blue-400" />}
          {title}
        </span>
          <span className="text-xs text-gray-600">{latency}s · {tokens} tokens</span>
        </div>
        <div className="p-4 text-xs text-gray-300 leading-relaxed whitespace-pre-wrap max-h-72 overflow-y-auto">{content}</div>
      </div>
  )
}

function ScoreBox({ label, value, color }: { label: string; value: number; color: string }) {
  const colors: Record<string, string> = { green: 'text-green-400', blue: 'text-blue-400', gray: 'text-gray-400' }
  return (
      <div className="text-center">
        <div className={`text-xl font-semibold ${colors[color]}`}>{(value * 100).toFixed(0)}%</div>
        <div className="text-xs text-gray-500 mt-1">{label}</div>
      </div>
  )
}

function ProfileField({ label, value, onChange }:
                      { label: string; value: number | ''; onChange: (v: number | '') => void }) {
  return (
      <div>
        <label className="text-xs text-gray-500 block mb-1 capitalize">{label}</label>
        <input
            type="number"
            value={value}
            onChange={e => onChange(e.target.value === '' ? '' : +e.target.value)}
            className="w-full bg-gray-800 border border-gray-700 rounded-md px-3 py-1.5 text-sm text-gray-200 focus:outline-none focus:border-blue-500"
        />
      </div>
  )
}

function InfoRow({ label, value }: { label: string; value: string }) {
  return (
      <div className="flex justify-between text-xs">
        <span className="text-gray-500">{label}</span>
        <span className="text-gray-300 text-right ml-4">{value}</span>
      </div>
  )
}