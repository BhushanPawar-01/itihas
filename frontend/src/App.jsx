import { useState } from 'react'
import QueryPanel     from './components/QueryPanel'
import NarrativePanel from './components/NarrativePanel'
import SidebarPanel   from './components/SidebarPanel'
import DebugPanel     from './components/DebugPanel'
import DebateFeed     from './components/DebateFeed'

/**
 * A single conversation turn stored in history.
 * Shape mirrors ConversationTurn on the backend (query.py).
 *
 * {
 *   query:         string,
 *   narrative:     string,
 *   source_chunks: Array | null,
 * }
 */

export default function App() {
  const [result, setResult]                   = useState(null)
  const [isLoading, setIsLoading]             = useState(false)
  const [error, setError]                     = useState(null)
  const [debateLog, setDebateLog]             = useState([])
  // Ordered list of prior completed turns — sent with every new request.
  // Each entry: { query, narrative, source_chunks }
  const [conversationHistory, setConversationHistory] = useState([])

  /**
   * Called by QueryPanel when a response arrives.
   * queryText is the raw query string from the input (passed back by QueryPanel).
   *
   * We append the completed turn to conversationHistory so the next request
   * carries full context. The turn shape is exactly what memory.py expects.
   */
  function handleResult(res, queryText) {
    setResult(res)
    setDebateLog([])

    setConversationHistory(prev => [
      ...prev,
      {
        query:         queryText,
        narrative:     res.narrative,
        source_chunks: res.source_chunks ?? null,
      },
    ])
  }

  /**
   * Clear the conversation — useful for starting a fresh session without
   * a page reload. Resets history, result, and debate log.
   */
  function handleNewConversation() {
    setConversationHistory([])
    setResult(null)
    setDebateLog([])
    setError(null)
  }

  const hasHistory = conversationHistory.length > 0

  return (
    <div className="min-h-screen bg-parchment">

      {/* Header */}
      <header className="bg-parchment-dark border-b border-parchment-dark px-6 py-4">
        <div className="max-w-[1400px] mx-auto flex items-baseline justify-between">
          <div className="flex items-baseline gap-4">
            <h1 className="font-samarkan text-3xl text-navy tracking-wide leading-none">
              Itihas
            </h1>
            <p className="text-xs text-navy-light">
              Adversarial AI · Indian military &amp; political history, 1600–1947
            </p>
          </div>

          {/* New conversation button — only visible once a turn exists */}
          {hasHistory && (
            <button
              onClick={handleNewConversation}
              className="text-xs text-navy-light hover:text-navy underline
                         underline-offset-2 transition-colors"
            >
              New conversation
            </button>
          )}
        </div>
      </header>

      <main className="max-w-[1400px] mx-auto px-4 py-8 space-y-8">

        {/* Prior turn summaries — shown above the query box once history exists */}
        {hasHistory && (
          <div className="w-full max-w-3xl mx-auto space-y-1">
            <p className="text-xs text-navy-light font-medium uppercase tracking-wide">
              Conversation so far
            </p>
            <div className="space-y-1">
              {conversationHistory.map((turn, i) => (
                <div
                  key={i}
                  className="flex items-start gap-2 text-xs text-navy-light
                             bg-white/60 rounded px-3 py-2 border border-parchment-dark"
                >
                  <span className="shrink-0 font-medium text-navy">
                    Q{i + 1}:
                  </span>
                  <span className="line-clamp-1">{turn.query}</span>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Query input */}
        <QueryPanel
          onResult={handleResult}
          isLoading={isLoading}
          setIsLoading={setIsLoading}
          error={error}
          setError={setError}
          onDebateStep={step => setDebateLog(prev => [...prev, step])}
          conversationHistory={conversationHistory}
        />

        {/* Live debate feed — visible while loading, hidden once result arrives */}
        {isLoading && (
          <DebateFeed steps={debateLog} />
        )}

        {/* Two-column result layout */}
        {result && !isLoading && (
          <div className="flex gap-0 items-start">

            {/* Left sidebar */}
            <SidebarPanel
              citations={result.citations}
              politicalAnalysis={result.political_analysis}
              militaryAnalysis={result.military_analysis}
            />

            {/* Right main column */}
            <div className="flex-1 min-w-0 pl-6 space-y-6">
              <NarrativePanel result={result} />
              <DebugPanel result={result} />
            </div>

          </div>
        )}

      </main>
    </div>
  )
}
