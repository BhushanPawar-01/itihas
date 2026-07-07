import { useState } from 'react'
import QueryPanel     from './components/QueryPanel'
import NarrativePanel from './components/NarrativePanel'
import SidebarPanel   from './components/SidebarPanel'
import DebugPanel     from './components/DebugPanel'
import DebateFeed     from './components/DebateFeed'

export default function App() {
  const [result, setResult]         = useState(null)
  const [isLoading, setIsLoading]   = useState(false)
  const [error, setError]           = useState(null)
  const [debateLog, setDebateLog]   = useState([])   // live agent steps during a query

  function handleResult(res) {
    setResult(res)
    setDebateLog([])   // clear live feed once final result arrives
  }

  return (
    <div className="min-h-screen bg-parchment">

      {/* Header — shares page palette */}
      <header className="bg-parchment-dark border-b border-parchment-dark px-6 py-4">
        <div className="max-w-[1400px] mx-auto flex items-baseline gap-4">
          <h1 className="font-samarkan text-3xl text-navy tracking-wide leading-none">
            Itihas
          </h1>
          <p className="text-xs text-navy-light">
            Adversarial AI · Indian military &amp; political history, 1600–1947
          </p>
        </div>
      </header>

      <main className="max-w-[1400px] mx-auto px-4 py-8 space-y-8">

        {/* Query input */}
        <QueryPanel
          onResult={handleResult}
          isLoading={isLoading}
          setIsLoading={setIsLoading}
          error={error}
          setError={setError}
          onDebateStep={step => setDebateLog(prev => [...prev, step])}
        />

        {/* Live debate feed — visible while loading, hidden once result arrives */}
        {isLoading && (
          <DebateFeed steps={debateLog} />
        )}

        {/* Two-column result layout */}
        {result && !isLoading && (
          <div className="flex gap-0 items-start">

            {/* Left sidebar — self-sizing, drag handle built in */}
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
