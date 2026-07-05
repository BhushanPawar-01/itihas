/**
 * NarrativePanel — renders the synthesised four-section narrative response.
 *
 * Props:
 *   result: QueryResponse object from the API
 *     result.narrative        : string (markdown with ## headings)
 *     result.confidence       : float 0.0–1.0
 *     result.citations        : string[] of doc_ids
 *     result.political_analysis : string
 *     result.military_analysis  : string
 *     result.critique_loops   : int
 */
import ReactMarkdown from 'react-markdown'

function ConfidenceBadge({ value }) {
  // Colour shifts green → yellow → red as confidence drops
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

export default function NarrativePanel({ result }) {
  if (!result) return null

  return (
    <div className="space-y-6">
      {/* Header row */}
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

      {/* Political and Military analysis — collapsible detail */}
      <details className="bg-white rounded-lg shadow-sm">
        <summary className="px-4 py-3 cursor-pointer text-sm font-medium text-navy-light
                            hover:text-navy select-none">
          Political analysis (raw agent output)
        </summary>
        <pre className="px-4 pb-4 text-xs text-gray-700 whitespace-pre-wrap">
          {result.political_analysis}
        </pre>
      </details>

      <details className="bg-white rounded-lg shadow-sm">
        <summary className="px-4 py-3 cursor-pointer text-sm font-medium text-navy-light
                            hover:text-navy select-none">
          Military analysis (raw agent output)
        </summary>
        <pre className="px-4 pb-4 text-xs text-gray-700 whitespace-pre-wrap">
          {result.military_analysis}
        </pre>
      </details>

      {/* Citations */}
      {result.citations.length > 0 && (
        <div>
          <h3 className="text-sm font-medium text-navy mb-2">Sources cited</h3>
          <div className="flex flex-wrap gap-2">
            {result.citations.map(id => (
              <span key={id}
                    className="px-2 py-0.5 bg-parchment-dark rounded text-xs font-mono text-navy">
                {id}
              </span>
            ))}
          </div>
        </div>
      )}
    </div>
  )
}