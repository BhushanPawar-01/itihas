/**
 * QueryPanel — query input, submit button, loading state, error display.
 *
 * Props:
 *   onResult(result)             — called with the full QueryResponse on success
 *   isLoading: bool
 *   setIsLoading(bool)
 *   error: string|null
 *   setError(string|null)
 *   onDebateStep(step)           — called with each SSE event from /query/stream
 *                                  while the query is in flight; feeds DebateFeed
 *   conversationHistory: Array   — ordered list of prior completed turns to send
 *                                  with every request; managed by App.jsx
 */
import { useRef, useState } from 'react'
import { submitQuery, streamQuery } from '../api/client'

export default function QueryPanel({
  onResult,
  isLoading,
  setIsLoading,
  error,
  setError,
  onDebateStep,
  conversationHistory = [],
}) {
  const [query, setQuery]               = useState('')
  const [includeDebug, setIncludeDebug] = useState(false)
  const submittingRef                   = useRef(false)
  const abortRef                        = useRef(null)

  async function handleSubmit() {
    if (!query.trim() || isLoading || submittingRef.current) return
    submittingRef.current = true
    setIsLoading(true)
    setError(null)

    abortRef.current?.abort()
    const controller = new AbortController()
    abortRef.current = controller

    try {
      const streamPromise = streamQuery(
        query.trim(),
        (event) => {
          if (event.type === 'node_complete' || event.type === 'rebuttal') {
            onDebateStep?.({
              agent:   event.agent,
              type:    event.type === 'rebuttal' ? 'loop' : 'output',
              content: event.content ?? '',
              label:   event.label  ?? '',
              loop:    event.loop   ?? 0,
            })
          }
        },
        controller.signal,
      ).catch(() => {
        // Stream errors are non-fatal — blocking result still comes through
      })

      const result = await submitQuery(query.trim(), includeDebug, conversationHistory)
      onResult(result, query.trim())   // pass query string so App can append to history

      await streamPromise
    } catch (err) {
      setError(err.message)
    } finally {
      submittingRef.current = false
      setIsLoading(false)
    }
  }

  function handleKeyDown(e) {
    if ((e.metaKey || e.ctrlKey) && e.key === 'Enter') handleSubmit()
  }

  return (
    <div className="w-full max-w-3xl mx-auto space-y-3">
      <label className="block text-sm font-medium text-navy-light">
        Historical query
      </label>
      <textarea
        className="w-full h-28 p-3 rounded-lg border border-parchment-dark
                   bg-white text-navy text-sm resize-none
                   focus:outline-none focus:ring-2 focus:ring-saffron"
        placeholder="e.g. Who commanded the INA at the Red Fort trials in 1945?"
        value={query}
        onChange={e => setQuery(e.target.value)}
        onKeyDown={handleKeyDown}
        disabled={isLoading}
      />
      <div className="flex items-center justify-between">
        <label className="flex items-center gap-2 text-sm text-navy-light cursor-pointer">
          <input
            type="checkbox"
            checked={includeDebug}
            onChange={e => setIncludeDebug(e.target.checked)}
            className="accent-saffron"
          />
          Show agent debug log
        </label>
        <button
          onClick={handleSubmit}
          disabled={!query.trim() || isLoading}
          className="px-5 py-2 rounded-lg bg-saffron text-white font-medium text-sm
                     hover:bg-saffron-dark disabled:opacity-40 disabled:cursor-not-allowed
                     transition-colors"
        >
          {isLoading ? 'Analysing…' : 'Analyse'}
        </button>
      </div>
      {error && (
        <p className="text-sm text-red-600 bg-red-50 rounded p-2">
          {error}
        </p>
      )}
    </div>
  )
}
