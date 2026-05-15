/**
 * canvasSimulation.js
 * Renders 200 Monte Carlo grey paths on an HTML5 Canvas element.
 * Used in Simulation page because Recharts cannot handle 200 simultaneous <Line> components.
 */

/**
 * Draw simulation paths on a canvas element.
 * @param {HTMLCanvasElement} canvas
 * @param {number[][]} paths   - array of arrays, shape [n_paths][n_days]
 * @param {object} theme       - { bg, surface, muted }
 */
export function drawSimulationPaths(canvas, paths, theme) {
  if (!canvas || !paths || paths.length === 0) return

  const ctx    = canvas.getContext('2d')
  const width  = canvas.width
  const height = canvas.height
  const nDays  = paths[0].length

  // Compute min/max across all paths for scaling
  let minVal = Infinity
  let maxVal = -Infinity
  for (const path of paths) {
    for (const v of path) {
      if (v < minVal) minVal = v
      if (v > maxVal) maxVal = v
    }
  }
  const range = maxVal - minVal || 1

  const xScale = (i) => (i / (nDays - 1)) * width
  const yScale = (v) => height - ((v - minVal) / range) * height * 0.9 - height * 0.05

  // Clear
  ctx.clearRect(0, 0, width, height)
  ctx.fillStyle = theme?.surface || '#0f1629'
  ctx.fillRect(0, 0, width, height)

  // Draw paths
  ctx.lineWidth = 0.8
  ctx.strokeStyle = 'rgba(148,163,184,0.06)'
  for (const path of paths) {
    ctx.beginPath()
    ctx.moveTo(xScale(0), yScale(path[0]))
    for (let i = 1; i < nDays; i++) {
      ctx.lineTo(xScale(i), yScale(path[i]))
    }
    ctx.stroke()
  }
}

/**
 * Draw percentile band labels on Y-axis of the canvas.
 */
export function drawYAxis(canvas, minVal, maxVal, theme) {
  if (!canvas) return
  const ctx    = canvas.getContext('2d')
  const width  = canvas.width
  const height = canvas.height
  const range  = maxVal - minVal || 1

  ctx.fillStyle   = theme?.muted || '#94a3b8'
  ctx.font        = '11px Inter, sans-serif'
  ctx.textAlign   = 'right'

  const ticks = 5
  for (let i = 0; i <= ticks; i++) {
    const v = minVal + (range / ticks) * i
    const y = height - ((v - minVal) / range) * height * 0.9 - height * 0.05
    ctx.fillText(`₹${Math.round(v)}`, width - 4, y + 4)
  }
}
