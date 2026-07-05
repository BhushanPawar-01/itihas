/**
 * Single entry point for all Itihas API calls.
 * No component imports fetch() directly.
 */

const BASE = '/api/v1'

/**
 * Submit a historical query to the agent graph.
 *
 * @param {string} query - The user's historical question.
 * @param {boolean} includeDebugLog - Whether to request the debug log.
 * @returns {Promise<Object>} - The full QueryResponse JSON.
 * @throws {Error} - Message is the API's detail string or a network error message.
 */
export async function submitQuery(query, includeDebugLog = false) {
  const response = await fetch(`${BASE}/query`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ query, include_debug_log: includeDebugLog }),
  })

  if (!response.ok) {
    const body = await response.json().catch(() => ({}))
    throw new Error(body.detail ?? `HTTP ${response.status}`)
  }

  return response.json()
}