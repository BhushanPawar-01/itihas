import { useState } from 'react'
import QueryPanel    from './components/QueryPanel'
import NarrativePanel from './components/NarrativePanel'
import SourcePanel   from './components/SourcePanel'
import DebugPanel from './components/DebugPanel'
import GraphPanel from './components/GraphPanel'

export default function App() {
  const [result, setResult]       = useState(null)
  const [isLoading, setIsLoading] = useState(false)
  const [error, setError]         = useState(null)

  return (
    <div className="min-h-screen bg-parchment">
      <header className="bg-navy py-4 px-6 shadow">
        <h1 className="text-2xl font-serif text-saffron tracking-wide">Itihas</h1>
        <p className="text-xs text-parchment-dark mt-0.5">
          Adversarial AI for Indian military and political history, 1600–1947
        </p>
      </header>

      <main className="max-w-6xl mx-auto px-4 py-8 space-y-8">
        <QueryPanel
          onResult={setResult}
          isLoading={isLoading}
          setIsLoading={setIsLoading}
          error={error}
          setError={setError}
        />
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          <div className="lg:col-span-2 space-y-6">
            <NarrativePanel result={result} />
          </div>
          <div className="lg:col-span-1 space-y-6">
            <SourcePanel chunks={result?.source_chunks} />
          </div>
        </div>
        <DebugPanel result={result} />
        <GraphPanel result={result} />
      </main>
    </div>
  )
}