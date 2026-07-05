/**
 * GraphPanel — D3 force-directed knowledge graph.
 *
 * Props:
 *   result: QueryResponse | null
 *
 * D3 owns the SVG. React owns only the wrapper div.
 * On result change: wipe SVG, rebuild graph.
 */
import { useEffect, useRef } from 'react'
import * as d3 from 'd3'
import { buildGraphData } from '../utils/buildGraphData'

const NODE_COLOURS = {
  document: '#1B2A4A',  // navy
  person:   '#FF9933',  // saffron
  event:    '#6B7280',  // gray
}

const NODE_RADIUS = {
  document: 8,
  person:   10,
  event:    7,
}

export default function GraphPanel({ result }) {
  const containerRef = useRef(null)

  useEffect(() => {
    const container = containerRef.current
    if (!container || !result) return

    // Wipe previous render
    d3.select(container).selectAll('*').remove()

    const { nodes, links } = buildGraphData(result)
    if (nodes.length === 0) return

    const width  = container.clientWidth || 600
    const height = 420

    const svg = d3.select(container)
      .append('svg')
      .attr('width', width)
      .attr('height', height)
      .attr('viewBox', `0 0 ${width} ${height}`)

    // Zoom + pan
    const g = svg.append('g')
    svg.call(
      d3.zoom()
        .scaleExtent([0.3, 3])
        .on('zoom', e => g.attr('transform', e.transform))
    )

    // Force simulation
    const sim = d3.forceSimulation(nodes)
      .force('link',   d3.forceLink(links).id(d => d.id).distance(80))
      .force('charge', d3.forceManyBody().strength(-120))
      .force('center', d3.forceCenter(width / 2, height / 2))
      .force('collide', d3.forceCollide(20))

    // Edges
    const link = g.append('g')
      .selectAll('line')
      .data(links)
      .join('line')
      .attr('stroke', '#CBD5E1')
      .attr('stroke-width', 1.5)

    // Nodes
    const node = g.append('g')
      .selectAll('circle')
      .data(nodes)
      .join('circle')
      .attr('r',    d => NODE_RADIUS[d.type] ?? 7)
      .attr('fill', d => NODE_COLOURS[d.type] ?? '#9CA3AF')
      .attr('cursor', 'pointer')
      .call(
        d3.drag()
          .on('start', (event, d) => {
            if (!event.active) sim.alphaTarget(0.3).restart()
            d.fx = d.x; d.fy = d.y
          })
          .on('drag', (event, d) => { d.fx = event.x; d.fy = event.y })
          .on('end', (event, d) => {
            if (!event.active) sim.alphaTarget(0)
            d.fx = null; d.fy = null
          })
      )

    // Labels
    const label = g.append('g')
      .selectAll('text')
      .data(nodes)
      .join('text')
      .text(d => d.label.length > 22 ? d.label.slice(0, 22) + '…' : d.label)
      .attr('font-size', 9)
      .attr('fill', '#374151')
      .attr('dx', d => (NODE_RADIUS[d.type] ?? 7) + 3)
      .attr('dy', '0.35em')

    // Tooltip
    const tooltip = d3.select(container)
      .append('div')
      .style('position', 'absolute')
      .style('background', 'white')
      .style('border', '1px solid #E5E7EB')
      .style('border-radius', '4px')
      .style('padding', '4px 8px')
      .style('font-size', '11px')
      .style('pointer-events', 'none')
      .style('opacity', 0)

    node
      .on('mouseover', (event, d) => {
        tooltip.transition().duration(150).style('opacity', 1)
        tooltip.html(`<strong>${d.type}</strong>: ${d.label}`)
          .style('left', `${event.offsetX + 12}px`)
          .style('top',  `${event.offsetY - 24}px`)
      })
      .on('mouseout', () => tooltip.transition().duration(150).style('opacity', 0))

    // Tick
    sim.on('tick', () => {
      link
        .attr('x1', d => d.source.x).attr('y1', d => d.source.y)
        .attr('x2', d => d.target.x).attr('y2', d => d.target.y)
      node
        .attr('cx', d => d.x).attr('cy', d => d.y)
      label
        .attr('x', d => d.x).attr('y', d => d.y)
    })

    // Cleanup on unmount or result change
    return () => sim.stop()
  }, [result])

  if (!result) return null

  return (
    <div className="space-y-3">
      <div className="flex items-center justify-between">
        <h2 className="text-lg font-serif font-bold text-navy">Knowledge Graph</h2>
        <div className="flex gap-3 text-xs text-gray-500">
          {[['document','navy'],['person','saffron'],['event','gray-500']].map(([t, c]) => (
            <span key={t} className="flex items-center gap-1">
              <span className={`w-2 h-2 rounded-full bg-${c}`} />
              {t}
            </span>
          ))}
        </div>
      </div>
      <div
        ref={containerRef}
        className="relative w-full rounded-lg border border-parchment-dark bg-white"
        style={{ height: '420px' }}
      />
    </div>
  )
}