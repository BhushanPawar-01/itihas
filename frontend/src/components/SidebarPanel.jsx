/**
 * SidebarPanel — resizable left sidebar.
 *
 * - Drag the right edge handle to resize between COLLAPSED_WIDTH and MAX_WIDTH.
 * - Below SNAP_THRESHOLD it collapses to icon-only strip (COLLAPSED_WIDTH).
 * - Three independently collapsible sections, default closed:
 *     1. References  — deduplicated title list, each linked to source URL
 *     2. Military Analysis — raw agent output via ReactMarkdown
 *     3. Political Analysis — raw agent output via ReactMarkdown
 *
 * Props:
 *   citations:         Array<{doc_id, title, url}>
 *   politicalAnalysis: string
 *   militaryAnalysis:  string
 */
import { useState, useRef, useEffect, useCallback } from 'react'
import ReactMarkdown from 'react-markdown'

const DEFAULT_WIDTH    = 280
const COLLAPSED_WIDTH  = 48
const SNAP_THRESHOLD   = 120   // px — below this, snap fully collapsed
const MAX_WIDTH        = 480

// ---------------------------------------------------------------------------
// Icons (inline SVG, no dependency)
// ---------------------------------------------------------------------------

const IconBook     = () => <svg viewBox="0 0 20 20" fill="currentColor" className="w-4 h-4"><path d="M9 4.804A7.968 7.968 0 005.5 4c-1.255 0-2.443.29-3.5.804v10A7.969 7.969 0 015.5 14c1.396 0 2.7.378 3.8 1.042A7.98 7.98 0 0114.5 14c1.255 0 2.443.29 3.5.804v-10A7.968 7.968 0 0014.5 4c-1.255 0-2.443.29-3.5.804V12a1 1 0 11-2 0V4.804z"/></svg>
const IconShield   = () => <svg viewBox="0 0 20 20" fill="currentColor" className="w-4 h-4"><path fillRule="evenodd" d="M10 1.944A11.954 11.954 0 012.166 5C2.056 5.649 2 6.319 2 7c0 5.225 3.34 9.67 8 11.317C14.66 16.67 18 12.225 18 7c0-.682-.057-1.35-.166-2.001A11.954 11.954 0 0110 1.944zM11 14a1 1 0 11-2 0 1 1 0 012 0zm0-7a1 1 0 10-2 0v3a1 1 0 102 0V7z" clipRule="evenodd"/></svg>
const IconFlag     = () => <svg viewBox="0 0 20 20" fill="currentColor" className="w-4 h-4"><path fillRule="evenodd" d="M3 6a3 3 0 013-3h10a1 1 0 01.8 1.6L14.25 8l2.55 3.4A1 1 0 0116 13H6a1 1 0 00-1 1v3a1 1 0 11-2 0V6z" clipRule="evenodd"/></svg>
const IconChevrons = ({ right }) => (
  <svg viewBox="0 0 20 20" fill="currentColor" className="w-3.5 h-3.5">
    {right
      ? <path fillRule="evenodd" d="M7.293 14.707a1 1 0 010-1.414L10.586 10 7.293 6.707a1 1 0 011.414-1.414l4 4a1 1 0 010 1.414l-4 4a1 1 0 01-1.414 0z" clipRule="evenodd"/>
      : <path fillRule="evenodd" d="M12.707 5.293a1 1 0 010 1.414L9.414 10l3.293 3.293a1 1 0 01-1.414 1.414l-4-4a1 1 0 010-1.414l4-4a1 1 0 011.414 0z" clipRule="evenodd"/>
    }
  </svg>
)

// ---------------------------------------------------------------------------
// CollapseSection
// ---------------------------------------------------------------------------

function CollapseSection({ title, icon, count, children, collapsed: sidebarCollapsed }) {
  const [open, setOpen] = useState(false)

  if (sidebarCollapsed) {
    // Icon-only strip — section toggle becomes a tooltip button
    return (
      <div className="relative group flex justify-center py-2">
        <button
          title={title}
          className="p-2 rounded text-navy-light hover:text-navy hover:bg-parchment transition-colors"
        >
          {icon}
        </button>
        {/* Tooltip */}
        <div className="absolute left-full ml-2 top-1/2 -translate-y-1/2 z-50
                        hidden group-hover:block bg-navy text-white text-xs
                        px-2 py-1 rounded whitespace-nowrap pointer-events-none">
          {title}
        </div>
      </div>
    )
  }

  return (
    <div className="rounded-lg border border-parchment-dark bg-white overflow-hidden">
      <button
        onClick={() => setOpen(o => !o)}
        className="w-full flex items-center justify-between px-3 py-2.5
                   text-sm font-medium text-navy hover:bg-parchment
                   transition-colors select-none text-left gap-2"
      >
        <span className="flex items-center gap-2 min-w-0">
          <span className="text-navy-light flex-shrink-0">{icon}</span>
          <span className="truncate">{title}</span>
          {count != null && count > 0 && (
            <span className="text-xs font-normal text-gray-400 flex-shrink-0">
              ({count})
            </span>
          )}
        </span>
        <span className="text-gray-400 text-xs flex-shrink-0">
          {open ? '▲' : '▼'}
        </span>
      </button>

      {open && (
        <div className="border-t border-parchment-dark">
          {children}
        </div>
      )}
    </div>
  )
}

// ---------------------------------------------------------------------------
// ReferencesList
// ---------------------------------------------------------------------------

function ReferencesList({ citations }) {
  if (!citations?.length) {
    return <p className="px-4 py-3 text-xs text-gray-400 italic">No sources available.</p>
  }

  const seen   = new Set()
  const unique = citations.filter(c => {
    const key = c.title || c.doc_id
    if (seen.has(key)) return false
    seen.add(key)
    return true
  })

  return (
    <div className="px-3 py-3 space-y-2 max-h-72 overflow-y-auto">
      {unique.map(c => (
        <div key={c.doc_id} className="flex items-start gap-1.5">
          <span className="mt-0.5 text-saffron text-xs flex-shrink-0">▸</span>
          {c.url ? (
            <a
              href={c.url}
              target="_blank"
              rel="noopener noreferrer"
              className="text-xs text-navy-light hover:text-saffron
                         underline underline-offset-2 leading-snug break-words"
            >
              {c.title || c.doc_id}
            </a>
          ) : (
            <span className="text-xs text-navy-light leading-snug break-words">
              {c.title || c.doc_id}
            </span>
          )}
        </div>
      ))}
    </div>
  )
}

// ---------------------------------------------------------------------------
// AgentMarkdown
// ---------------------------------------------------------------------------

function AgentMarkdown({ content }) {
  if (!content) {
    return <p className="px-3 py-3 text-xs text-gray-400 italic">No output available.</p>
  }
  return (
    <div className="px-3 py-3 max-h-96 overflow-y-auto
                    prose prose-xs max-w-none
                    prose-headings:text-navy prose-headings:font-serif
                    prose-headings:text-sm prose-headings:font-semibold
                    prose-p:text-xs prose-p:text-gray-700 prose-p:leading-relaxed
                    prose-strong:text-navy">
      <ReactMarkdown>{content}</ReactMarkdown>
    </div>
  )
}

// ---------------------------------------------------------------------------
// Main export
// ---------------------------------------------------------------------------

export default function SidebarPanel({ citations, politicalAnalysis, militaryAnalysis }) {
  const [width, setWidth]           = useState(DEFAULT_WIDTH)
  const [isCollapsed, setCollapsed] = useState(false)
  const dragging                    = useRef(false)
  const startX                      = useRef(0)
  const startWidth                  = useRef(DEFAULT_WIDTH)

  const onMouseDown = useCallback(e => {
    e.preventDefault()
    dragging.current  = true
    startX.current    = e.clientX
    startWidth.current = width
    document.body.style.cursor    = 'col-resize'
    document.body.style.userSelect = 'none'
  }, [width])

  useEffect(() => {
    function onMouseMove(e) {
      if (!dragging.current) return
      const delta    = e.clientX - startX.current
      const newWidth = Math.min(MAX_WIDTH, Math.max(COLLAPSED_WIDTH, startWidth.current + delta))

      if (newWidth <= SNAP_THRESHOLD) {
        setCollapsed(true)
        setWidth(COLLAPSED_WIDTH)
      } else {
        setCollapsed(false)
        setWidth(newWidth)
      }
    }

    function onMouseUp() {
      if (!dragging.current) return
      dragging.current = false
      document.body.style.cursor     = ''
      document.body.style.userSelect = ''
    }

    window.addEventListener('mousemove', onMouseMove)
    window.addEventListener('mouseup',   onMouseUp)
    return () => {
      window.removeEventListener('mousemove', onMouseMove)
      window.removeEventListener('mouseup',   onMouseUp)
    }
  }, [])

  function toggleCollapse() {
    if (isCollapsed) {
      setCollapsed(false)
      setWidth(DEFAULT_WIDTH)
    } else {
      setCollapsed(true)
      setWidth(COLLAPSED_WIDTH)
    }
  }

  const citationCount = (() => {
    if (!citations?.length) return 0
    const seen = new Set()
    citations.forEach(c => seen.add(c.title || c.doc_id))
    return seen.size
  })()

  return (
    <div
      className="relative flex-shrink-0 flex"
      style={{ width: isCollapsed ? COLLAPSED_WIDTH : width }}
    >
      {/* Sidebar body */}
      <div className="flex-1 min-w-0 overflow-hidden">

        {/* Top bar — collapse toggle */}
        <div className={`flex items-center mb-2 ${isCollapsed ? 'justify-center' : 'justify-end px-1'}`}>
          <button
            onClick={toggleCollapse}
            title={isCollapsed ? 'Expand sidebar' : 'Collapse sidebar'}
            className="p-1.5 rounded text-navy-light hover:text-navy
                       hover:bg-parchment-dark transition-colors"
          >
            <IconChevrons right={isCollapsed} />
          </button>
        </div>

        <div className={isCollapsed ? 'space-y-1' : 'space-y-2'}>
          <CollapseSection
            title="References"
            icon={<IconBook />}
            count={citationCount}
            collapsed={isCollapsed}
          >
            <ReferencesList citations={citations} />
          </CollapseSection>

          <CollapseSection
            title="Military Analysis"
            icon={<IconShield />}
            collapsed={isCollapsed}
          >
            <AgentMarkdown content={militaryAnalysis} />
          </CollapseSection>

          <CollapseSection
            title="Political Analysis"
            icon={<IconFlag />}
            collapsed={isCollapsed}
          >
            <AgentMarkdown content={politicalAnalysis} />
          </CollapseSection>
        </div>
      </div>

      {/* Drag handle — right edge */}
      {!isCollapsed && (
        <div
          onMouseDown={onMouseDown}
          className="absolute right-0 top-0 bottom-0 w-1.5 cursor-col-resize
                     hover:bg-saffron/30 active:bg-saffron/50 transition-colors
                     rounded-full"
          title="Drag to resize"
        />
      )}
    </div>
  )
}
