/**
 * Extract nodes and edges from a QueryResponse for the D3 graph.
 *
 * Node types:
 *   document — a source doc_id (from citations)
 *   person   — named entity extracted from narrative text (heuristic: Title Case words
 *              preceded by known honorifics or military ranks)
 *   event    — INA Trials, Battle of X, Siege of X (heuristic: "Battle of", "Siege of",
 *              "Trial", "Uprising" in narrative)
 *
 * Edges:
 *   document → person   (doc_id mentions person name — substring match in chunk text)
 *   document → event    (doc_id mentions event — substring match)
 *   person   → event    (person name appears in same chunk as event name)
 *
 * This is heuristic, not NER. It will miss things and produce false positives.
 * Good enough for a demo graph — the agent system is the intelligence layer, not this function.
 *
 * @param {Object} result - Full QueryResponse from the API
 * @returns {{ nodes: Array, links: Array }}
 *   nodes: [{ id, label, type }]
 *   links: [{ source, target, type }]
 */

const HONORIFICS = ['General', 'Colonel', 'Major', 'Captain', 'Lieutenant',
                    'Shah', 'Nawaz', 'Khan', 'Bose', 'Subhas', 'Nehru',
                    'Gandhi', 'Dhillon', 'Sahgal', 'Singh', 'Rani', 'Lakshmi']

const EVENT_TRIGGERS = ['Battle of', 'Siege of', 'Trial', 'Uprising',
                        'Mutiny', 'Campaign', 'Operation', 'March']

function extractPersons(text) {
  const found = new Set()
  HONORIFICS.forEach(h => {
    const re = new RegExp(`${h}\\s+[A-Z][a-z]+(?:\\s+[A-Z][a-z]+)*`, 'g')
    const matches = text.match(re) ?? []
    matches.forEach(m => found.add(m.trim()))
    // Also add the honorific alone if it appears as a standalone name
    if (text.includes(h)) found.add(h)
  })
  return [...found]
}

function extractEvents(text) {
  const found = new Set()
  EVENT_TRIGGERS.forEach(trigger => {
    const re = new RegExp(`${trigger}\\s+[A-Za-z\\s]{2,30}`, 'g')
    const matches = text.match(re) ?? []
    matches.forEach(m => found.add(m.trim().slice(0, 40)))
  })
  return [...found]
}

export function buildGraphData(result) {
  if (!result) return { nodes: [], links: [] }

  const narrative = result.narrative ?? ''
  const chunks    = result.source_chunks ?? []

  const persons = extractPersons(narrative)
  const events  = extractEvents(narrative)
  const docIds  = result.citations ?? []

  // Build node list — deduplicate by id
  const nodeMap = new Map()

  docIds.forEach(id => {
    if (!nodeMap.has(id))
      nodeMap.set(id, { id, label: id.slice(0, 20), type: 'document' })
  })
  persons.forEach(p => {
    const id = `person:${p}`
    if (!nodeMap.has(id))
      nodeMap.set(id, { id, label: p, type: 'person' })
  })
  events.forEach(e => {
    const id = `event:${e}`
    if (!nodeMap.has(id))
      nodeMap.set(id, { id, label: e, type: 'event' })
  })

  // Build edges — chunk text as bridge
  const links = []
  const seen  = new Set()

  function addLink(source, target, type) {
    const key = `${source}||${target}`
    if (!seen.has(key) && nodeMap.has(source) && nodeMap.has(target)) {
      seen.add(key)
      links.push({ source, target, type })
    }
  }

  chunks.forEach(chunk => {
    const text = chunk.text ?? ''
    persons.forEach(p => {
      if (text.includes(p)) addLink(chunk.doc_id, `person:${p}`, 'mentions')
    })
    events.forEach(e => {
      const trigger = e.split(' ').slice(0, 2).join(' ')
      if (text.includes(trigger)) addLink(chunk.doc_id, `event:${e}`, 'mentions')
    })
  })

  // Person → event edges from narrative co-occurrence
  persons.forEach(p => {
    events.forEach(e => {
      const pIdx = narrative.indexOf(p)
      const eIdx = narrative.indexOf(e.split(' ').slice(0, 2).join(' '))
      if (pIdx !== -1 && eIdx !== -1 && Math.abs(pIdx - eIdx) < 300) {
        addLink(`person:${p}`, `event:${e}`, 'associated')
      }
    })
  })

  return { nodes: [...nodeMap.values()], links }
}