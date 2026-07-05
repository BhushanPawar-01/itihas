/**
 * DebugPanel — agent pipeline debug view.
 *
 * Props:
 *   result: QueryResponse — full API result
 *
 * Renders only if result.debug_log is a non-empty array.
 * User must have checked "Show agent debug log" in QueryPanel.
 */
import { useState } from 'react'

const AGENT_COLOURS = {
  source_agent:    'border-blue-400   bg-blue-50',
  political_agent: 'border-orange-400 bg-orange-50',
  military_agent:  'border-green-400  bg-green-50',
  critique_agent:  'border-purple-400 bg-purple-50',
  narrative_agent: 'border-saffron    bg-yellow-50',
}

function agentColour(logLine) {
  for (const [key, cls] of Object.entries(AGENT_COLOURS)) {
    if (logLine.startsWith(key.replace('_agent', '_agent'))) return cls
  }
  return 'border-gray-300 bg-gray-50'
}

function Timeline({ logs }) {
  return (
    <ol className="relative border-l border-gray-200 space-y-4 pl-4">
      {logs.map((entry, i) => (
        <li key={i} className="relative">
          <span className="absolute -left-[1.15rem] flex h-4 w-4 items-center justify-center
                           rounded-full bg-saffron ring-2 ring-white text-[9px] text-white font-bold">
            {i + 1}
          </span>
          <p className={`ml-2 rounded border-l-4 px-3 py-2 text-xs font-mono
                         ${agentColour(entry)}`}>
            {entry}
          </p>
        </li>
      ))}
    </ol>
  )
}

function PerAgent({ result }) {
  const sections = [
    { label: 'Political Agent',  content: result.political_analysis,  colour: 'text-orange-700' },
    { label: 'Military Agent',   content: result.military_analysis,   colour: 'text-green-700'  },
    { label: 'Critique Agent',   content: result.critique_output,     colour: 'text-purple-700' },
  ].filter(s => s.content)

  return (
    <div className="space-y-3">
      {sections.map(({ label, content, colour }) => (
        <details key={label} className="bg-white rounded-lg shadow-sm">
          <summary className={`px-4 py-3 cursor-pointer text-sm font-medium
                               hover:opacity-80 select-none ${colour}`}>
            {label}
          </summary>
          <pre className="px-4 pb-4 text-xs text-gray-700 whitespace-pre-wrap overflow-auto max-h-64">
            {content}
          </pre>
        </details>
      ))}
    </div>
  )
}

export default function DebugPanel({ result }) {
  const [tab, setTab] = useState('timeline')

  if (!result?.debug_log?.length) return null

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <h2 className="text-lg font-serif font-bold text-navy">Agent Debug</h2>
        <span className="text-xs text-gray-500">
          {result.critique_loops} critique loop{result.critique_loops !== 1 ? 's' : ''}
        </span>
      </div>

      {/* Tab bar */}
      <div className="flex gap-2 border-b border-parchment-dark">
        {['timeline', 'per-agent'].map(t => (
          <button
            key={t}
            onClick={() => setTab(t)}
            className={`px-3 py-1.5 text-sm capitalize transition-colors
              ${tab === t
                ? 'border-b-2 border-saffron text-saffron font-medium'
                : 'text-gray-500 hover:text-navy'}`}
          >
            {t.replace('-', ' ')}
          </button>
        ))}
      </div>

      {tab === 'timeline'  && <Timeline logs={result.debug_log} />}
      {tab === 'per-agent' && <PerAgent result={result} />}
    </div>
  )
}