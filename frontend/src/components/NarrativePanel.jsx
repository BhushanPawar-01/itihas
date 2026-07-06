import { useState } from 'react'
import ReactMarkdown from 'react-markdown'

function ConfidenceBadge({ value }) {
  const pct = Math.round(value * 100)
  const colour = pct >= 75 ? 'bg-green-100 text-green-800'
               : pct >= 50 ? 'bg-yellow-100 text-yellow-800'
               :              'bg-red-100 text-red-800'
  return (
    <span className={`inline-block px-2 py-0.5 rounded text-xs font-medium ${colour}`}>
      Confidence: {pct}%
    </span>
  )
}

function CollapsibleSection({ title, children }) {
  const [open, setOpen] = useState(false)
  return (
    <div className="bg-white rounded-lg shadow-sm overflow-hidden">
      <button
        onClick={() => setOpen(o => !o)}
        className="w-full flex items-center justify-between px-4 py-3
                   text-sm font-medium text-navy-light hover:text-navy
                   hover:bg-parchment transition-colors select-none text-left"
      >
        <span>{title}</span>
        <span className="text-gray-400 text-xs ml-2">{open ? '▲ collapse' : '▼ expand'}</span>
      </button>
      {open && (
        <pre className="px-4 pb-4 pt-1 text-xs text-gray-700 whitespace-pre-wrap border-t border-parchment-dark">
          {children}
        </pre>
      )}
    </div>
  )
}

export default function NarrativePanel({ result }) {
  if (!result) return null

  return (
    <div className="space-y-6">

      {/* Header */}
      <div className="flex items-center justify-between">
        <h2 className="text-lg font-serif font-bold text-navy">Analysis</h2>
        <div className="flex items-center gap-3">
          <ConfidenceBadge value={result.confidence} />
          <span className="text-xs text-gray-500">
            {result.critique_loops} critique loop{result.critique_loops !== 1 ? 's' : ''}
          </span>
        </div>
      </div>

      {/* Narrative — four ## sections rendered as markdown */}
      <div className="prose prose-sm max-w-none bg-white rounded-lg p-6 shadow-sm
                      prose-headings:text-navy prose-headings:font-serif">
        <ReactMarkdown>{result.narrative}</ReactMarkdown>
      </div>

      {/* Political and Military — collapsed by default, toggle on click */}
      <CollapsibleSection title="Political analysis (raw agent output)">
        {result.political_analysis}
      </CollapsibleSection>

      <CollapsibleSection title="Military analysis (raw agent output)">
        {result.military_analysis}
      </CollapsibleSection>

      {/* Citations — unique titles with source links */}
      {result.citations?.length > 0 && (
        <div className="bg-white rounded-lg shadow-sm p-4">
          <h3 className="text-sm font-medium text-navy mb-3">Sources cited</h3>
          <div className="flex flex-col gap-2">
            {result.citations.map(c => (
              <div key={c.doc_id} className="flex items-start gap-2">
                <span className="mt-0.5 text-saffron text-xs flex-shrink-0">▸</span>
                {c.url ? (
                  <a
                    href={c.url}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="text-sm text-navy hover:text-saffron underline underline-offset-2"
                  >
                    {c.title || c.doc_id}
                  </a>
                ) : (
                  <span className="text-sm text-navy font-mono">
                    {c.title || c.doc_id}
                  </span>
                )}
              </div>
            ))}
          </div>
        </div>
      )}

    </div>
  )
}