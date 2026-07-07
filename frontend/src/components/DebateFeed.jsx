/**
 * DebateFeed — live agent debate display shown while a query is in flight.
 *
 * Props:
 *   steps: Array<{agent: string, type: string, content: string}>
 *          Each step is pushed by QueryPanel via onDebateStep() as SSE
 *          events arrive from the backend. Empty array = loading state.
 *
 * Agent types and what they mean:
 *   source_agent    — retrieval underway
 *   political_agent — political analysis pass (or rebuttal)
 *   military_agent  — military analysis pass (or rebuttal)
 *   critique_agent  — contradiction check
 *   narrative_agent — final synthesis
 *
 * step.type values:
 *   "start"    — agent just began
 *   "output"   — partial or complete agent output
 *   "loop"     — critique triggered a rebuttal loop
 *   "done"     — agent finished
 */

const AGENT_META = {
  source_agent:    { label: 'Source',    accent: 'border-blue-400',   bg: 'bg-blue-50',    dot: 'bg-blue-400'   },
  political_agent: { label: 'Political', accent: 'border-orange-400', bg: 'bg-orange-50',  dot: 'bg-orange-400' },
  military_agent:  { label: 'Military',  accent: 'border-green-400',  bg: 'bg-green-50',   dot: 'bg-green-400'  },
  critique_agent:  { label: 'Critique',  accent: 'border-purple-400', bg: 'bg-purple-50',  dot: 'bg-purple-400' },
  narrative_agent: { label: 'Narrative', accent: 'border-saffron',    bg: 'bg-yellow-50',  dot: 'bg-saffron'    },
}

const DEFAULT_META = { label: 'Agent', accent: 'border-gray-300', bg: 'bg-gray-50', dot: 'bg-gray-400' }

function getMeta(agent) {
  return AGENT_META[agent] ?? DEFAULT_META
}

function AgentDot({ agent, pulse = false }) {
  const meta = getMeta(agent)
  return (
    <span className={`inline-block w-2 h-2 rounded-full flex-shrink-0 mt-1.5
                      ${meta.dot} ${pulse ? 'animate-pulse' : ''}`} />
  )
}

function StepRow({ step, isLast }) {
  const meta = getMeta(step.agent)
  const isLoop = step.type === 'loop'

  return (
    <div className="flex gap-3">
      <div className="flex flex-col items-center gap-0">
        <AgentDot agent={step.agent} pulse={isLast} />
        {!isLast && <div className="w-px flex-1 bg-parchment-dark mt-1" />}
      </div>

      <div className={`flex-1 mb-3 rounded-md border-l-4 px-3 py-2
                       ${meta.accent} ${meta.bg}
                       ${isLoop ? 'ring-1 ring-purple-300' : ''}`}>
        <div className="flex items-center gap-2 mb-1">
          <span className="text-xs font-semibold text-navy-light uppercase tracking-wide">
            {meta.label}
          </span>
          {isLoop && (
            <span className="text-xs text-purple-700 font-medium">
              ↺ rebuttal loop
            </span>
          )}
          {step.type === 'done' && (
            <span className="text-xs text-gray-400">✓</span>
          )}
        </div>
        {step.content && (
          <p className="text-xs text-gray-700 leading-relaxed whitespace-pre-wrap">
            {step.content}
          </p>
        )}
      </div>
    </div>
  )
}

function ThinkingSpinner() {
  // Shown when steps array is empty — agents haven't emitted anything yet
  const agents = ['source_agent', 'political_agent', 'military_agent', 'critique_agent', 'narrative_agent']

  return (
    <div className="space-y-3">
      <p className="text-xs text-navy-light font-medium uppercase tracking-wide mb-4">
        Agents initialising…
      </p>
      {agents.map((agent, i) => {
        const meta = getMeta(agent)
        return (
          <div key={agent} className="flex gap-3 items-center opacity-30"
               style={{ animationDelay: `${i * 150}ms` }}>
            <span className={`w-2 h-2 rounded-full flex-shrink-0 ${meta.dot}`} />
            <div className={`flex-1 h-6 rounded ${meta.bg} border-l-4 ${meta.accent}`} />
          </div>
        )
      })}
    </div>
  )
}

export default function DebateFeed({ steps }) {
  if (!steps?.length) {
    return (
      <div className="max-w-2xl mx-auto bg-white rounded-lg border border-parchment-dark p-6">
        <ThinkingSpinner />
      </div>
    )
  }

  return (
    <div className="max-w-2xl mx-auto bg-white rounded-lg border border-parchment-dark p-6">
      <p className="text-xs text-navy-light font-medium uppercase tracking-wide mb-4">
        Agent debate — live
      </p>
      <div>
        {steps.map((step, i) => (
          <StepRow key={i} step={step} isLast={i === steps.length - 1} />
        ))}
      </div>
    </div>
  )
}
