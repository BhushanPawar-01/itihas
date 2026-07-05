/**
 * SourcePanel — retrieved evidence chunks grouped by bias_tag.
 *
 * Props:
 *   chunks: list of {doc_id, chunk_index, text, bias_tag, score}
 *           from result.source_chunks
 */
import { useState } from 'react'
import { getBiasTagMeta } from '../constants/biasTags'

function ChunkCard({ chunk }) {
  const [expanded, setExpanded] = useState(false)
  const meta = getBiasTagMeta(chunk.bias_tag)

  return (
    <div className="bg-white rounded-lg shadow-sm p-4 space-y-2">
      <div className="flex items-center justify-between">
        <span className={`px-2 py-0.5 rounded text-xs font-medium ${meta.bg} ${meta.text}`}>
          {meta.label}
        </span>
        <span className="text-xs font-mono text-gray-400">
          {chunk.doc_id} #{chunk.chunk_index}
        </span>
      </div>
      <p className="text-sm text-navy leading-relaxed">
        {expanded ? chunk.text : `${chunk.text.slice(0, 200)}…`}
      </p>
      {chunk.text.length > 200 && (
        <button
          onClick={() => setExpanded(e => !e)}
          className="text-xs text-saffron hover:text-saffron-dark"
        >
          {expanded ? 'Show less' : 'Show more'}
        </button>
      )}
      <div className="text-xs text-gray-400">
        RRF score: {chunk.score.toFixed(4)}
      </div>
    </div>
  )
}

export default function SourcePanel({ chunks }) {
  if (!chunks || chunks.length === 0) return null

  // Group by bias_tag, sorted alphabetically for determinism
  const groups = chunks.reduce((acc, chunk) => {
    acc[chunk.bias_tag] = acc[chunk.bias_tag] ?? []
    acc[chunk.bias_tag].push(chunk)
    return acc
  }, {})

  return (
    <div className="space-y-4">
      <h2 className="text-lg font-serif font-bold text-navy">
        Retrieved Evidence ({chunks.length} chunks)
      </h2>
      {Object.keys(groups).sort().map(tag => {
        const meta = getBiasTagMeta(tag)
        return (
          <details key={tag} open>
            <summary className={`px-3 py-2 rounded cursor-pointer select-none
                                 text-sm font-medium ${meta.bg} ${meta.text}`}>
              {meta.label} — {groups[tag].length} chunk{groups[tag].length !== 1 ? 's' : ''}
            </summary>
            <div className="mt-2 space-y-2 pl-2">
              {groups[tag].map(chunk => (
                <ChunkCard key={`${chunk.doc_id}-${chunk.chunk_index}`} chunk={chunk} />
              ))}
            </div>
          </details>
        )
      })}
    </div>
  )
}