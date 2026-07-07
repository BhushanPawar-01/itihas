/**
 * NarrativePanel — main result column.
 *
 * Parses the narrative string into up to four named sections delimited by ##
 * headings and renders each as its own card, rather than one continuous
 * markdown blob.
 *
 * Inline citation rewrite: [doc_id] markers in the narrative text are replaced
 * with hyperlinked title fragments using the citations array already returned
 * by the API. Pure frontend transform — no backend change needed.
 *
 * Props:
 *   result: QueryResponse
 */
import ReactMarkdown from 'react-markdown'

// ---------------------------------------------------------------------------
// Citation rewriting
// ---------------------------------------------------------------------------

/**
 * Replace every [doc_id] marker in text with a linked title fragment.
 * Returns a string — ReactMarkdown will render the injected <a> tags
 * because we pass it through rehype (raw HTML disabled by default in
 * react-markdown v10, so we use a custom component approach instead:
 * replace markers with markdown link syntax before parsing).
 *
 * Strategy: swap [doc_id] → [Short Title](url) or just **Short Title**
 * if there's no URL. ReactMarkdown then renders these as real links.
 */
function rewriteCitations(text, citations) {
  if (!citations?.length || !text) return text

  const byDocId = Object.fromEntries(citations.map(c => [c.doc_id, c]))

  return text.replace(/\[([^\]]+)\]/g, (match, docId) => {
    const citation = byDocId[docId]
    if (!citation) return match                         // not a known doc_id, leave as-is

    const label = citation.title
      ? citation.title.length > 40
        ? citation.title.slice(0, 40) + '…'
        : citation.title
      : docId

    return citation.url
      ? `[${label}](${citation.url})`
      : `**${label}**`
  })
}

// ---------------------------------------------------------------------------
// Section parsing
// ---------------------------------------------------------------------------

const SECTION_ORDER = [
  'Political Reality',
  'Military Reality',
  'Ground Truth vs Propaganda',
  'Confidence Assessment',
]

/**
 * Split the narrative string on ## headings.
 * Returns an array of { heading: string, body: string }.
 * If no ## headings found, returns a single entry with heading=null and body=full text.
 */
function parseSections(narrative) {
  if (!narrative) return []

  const lines    = narrative.split('\n')
  const sections = []
  let current    = null

  for (const line of lines) {
    const headingMatch = line.match(/^#{1,3}\s+(.+)$/)
    if (headingMatch) {
      if (current) sections.push(current)
      current = { heading: headingMatch[1].trim(), body: '' }
    } else {
      if (!current) current = { heading: null, body: '' }
      current.body += line + '\n'
    }
  }
  if (current) sections.push(current)

  return sections.filter(s => s.body.trim() || s.heading)
}

// ---------------------------------------------------------------------------
// Section card colours — keyed on heading content
// ---------------------------------------------------------------------------

function sectionAccent(heading) {
  if (!heading) return { border: 'border-parchment-dark', label: 'text-navy' }
  const h = heading.toLowerCase()
  if (h.includes('political'))   return { border: 'border-orange-300', label: 'text-orange-800' }
  if (h.includes('military'))    return { border: 'border-green-300',  label: 'text-green-800'  }
  if (h.includes('propaganda') || h.includes('ground truth'))
                                 return { border: 'border-purple-300', label: 'text-purple-800' }
  if (h.includes('confidence'))  return { border: 'border-blue-300',   label: 'text-blue-800'   }
  return { border: 'border-parchment-dark', label: 'text-navy' }
}

// ---------------------------------------------------------------------------
// Confidence badge
// ---------------------------------------------------------------------------

function ConfidenceBadge({ value }) {
  if (value == null) return null
  const pct    = Math.round(value * 100)
  const colour = pct >= 75 ? 'bg-green-100 text-green-800'
               : pct >= 50 ? 'bg-yellow-100 text-yellow-800'
               :              'bg-red-100 text-red-800'
  return (
    <span className={`inline-block px-2 py-0.5 rounded text-xs font-medium ${colour}`}>
      {pct}% confidence
    </span>
  )
}

// ---------------------------------------------------------------------------
// Main component
// ---------------------------------------------------------------------------

export default function NarrativePanel({ result }) {
  if (!result) return null

  const rewritten = rewriteCitations(result.narrative, result.citations)
  const sections  = parseSections(rewritten)

  return (
    <div className="space-y-4">

      {/* Panel header */}
      <div className="flex items-center justify-between">
        <h2 className="text-lg font-serif font-bold text-navy">Analysis</h2>
        <div className="flex items-center gap-3">
          <ConfidenceBadge value={result.confidence} />
          {result.critique_loops != null && (
            <span className="text-xs text-gray-400">
              {result.critique_loops} critique loop{result.critique_loops !== 1 ? 's' : ''}
            </span>
          )}
        </div>
      </div>

      {/* Section cards */}
      {sections.length > 0 ? (
        sections.map((section, i) => {
          const accent = sectionAccent(section.heading)
          return (
            <div
              key={i}
              className={`bg-white rounded-lg shadow-sm border-l-4 ${accent.border} overflow-hidden`}
            >
              {section.heading && (
                <div className="px-5 pt-4 pb-1">
                  <h3 className={`text-sm font-semibold font-serif ${accent.label}`}>
                    {section.heading}
                  </h3>
                </div>
              )}
              <div className="px-5 py-4 prose prose-sm max-w-none
                              prose-headings:text-navy prose-headings:font-serif
                              prose-a:text-saffron prose-a:no-underline
                              hover:prose-a:underline
                              prose-strong:text-navy">
                <ReactMarkdown>{section.body}</ReactMarkdown>
              </div>
            </div>
          )
        })
      ) : (
        /* Fallback: render as a single card if parsing yields nothing */
        <div className="bg-white rounded-lg shadow-sm p-6
                        prose prose-sm max-w-none
                        prose-headings:text-navy prose-headings:font-serif
                        prose-a:text-saffron">
          <ReactMarkdown>{rewritten}</ReactMarkdown>
        </div>
      )}

    </div>
  )
}
