/**
 * Single entry point for all Itihas API calls.
 * No component imports fetch() directly.
 */

const BASE = '/api/v1'

/**
 * Submit a historical query to the agent graph (blocking).
 * Returns the full QueryResponse JSON.
 *
 * @param {string}  query
 * @param {boolean} includeDebugLog
 * @returns {Promise<Object>}
 * @throws {Error}
 */
export async function submitQuery(query, includeDebugLog = false) {
  const response = await fetch(`${BASE}/query`, {
    method:  'POST',
    headers: { 'Content-Type': 'application/json' },
    body:    JSON.stringify({ query, include_debug_log: includeDebugLog }),
  })

  if (!response.ok) {
    const body = await response.json().catch(() => ({}))
    throw new Error(body.detail ?? `HTTP ${response.status}`)
  }

  return response.json()
}

/**
 * Stream agent debate events while a query runs.
 *
 * Opens POST /query/stream and reads the SSE response line by line.
 * Calls onEvent(parsedPayload) for each event received.
 * Resolves when the stream closes (type="done" or type="error").
 *
 * Event payload shapes (see backend/routes/query.py for full spec):
 *   { type: "node_complete" | "rebuttal", agent, label, loop, content, error }
 *   { type: "done" }
 *   { type: "error", content, traceback }
 *
 * @param {string}   query
 * @param {Function} onEvent  — called with each parsed event object
 * @param {AbortSignal} [signal] — optional AbortSignal to cancel the stream
 * @returns {Promise<void>}
 */
export async function streamQuery(query, onEvent, signal) {
  const response = await fetch(`${BASE}/query/stream`, {
    method:  'POST',
    headers: { 'Content-Type': 'application/json' },
    body:    JSON.stringify({ query }),
    signal,
  })

  if (!response.ok) {
    const body = await response.json().catch(() => ({}))
    throw new Error(body.detail ?? `HTTP ${response.status}`)
  }

  const reader  = response.body.getReader()
  const decoder = new TextDecoder()
  let   buffer  = ''

  while (true) {
    const { done, value } = await reader.read()
    if (done) break

    buffer += decoder.decode(value, { stream: true })

    // SSE messages are separated by double newlines
    const messages = buffer.split('\n\n')
    buffer = messages.pop()   // last element may be incomplete — keep in buffer

    for (const message of messages) {
      const line = message.trim()
      if (!line.startsWith('data:')) continue

      const jsonStr = line.slice('data:'.length).trim()
      if (!jsonStr) continue

      try {
        const payload = JSON.parse(jsonStr)
        onEvent(payload)
        if (payload.type === 'done' || payload.type === 'error') return
      } catch {
        // Malformed JSON in SSE line — skip silently
      }
    }
  }
}