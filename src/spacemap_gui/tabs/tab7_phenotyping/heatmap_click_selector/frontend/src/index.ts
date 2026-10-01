import { RenderData, Streamlit } from "streamlit-component-lib"
import plotlyUrl from "plotly.js-dist-min/plotly.min.js?url"

declare global {
  interface Window {
    Plotly?: any
  }
}

const root = document.body.appendChild(document.createElement("div"))
root.id = "heatmap-click-selector-root"

const style = document.head.appendChild(document.createElement("style"))
style.textContent = `
  :root {
    --bg: #0e1117;
    --text: #fafafa;
    --muted: rgba(250, 250, 250, 0.72);
    --border: rgba(250, 250, 250, 0.18);
    --surface: rgba(255, 255, 255, 0.03);
    --primary: #ff4b4b;
    --font: sans-serif;
  }

  html, body {
    margin: 0;
    padding: 0;
    width: 100%;
    background: var(--bg);
    color: var(--text);
    font-family: var(--font);
  }

  #heatmap-click-selector-root {
    width: 100%;
    background: var(--bg);
    color: var(--text);
  }

  #plotly-heatmap-container {
    width: 100%;
  }

  .selector-panel {
    padding-top: 0.5rem;
  }

  .selector-caption {
    margin: 0 0 0.6rem 0;
    color: var(--muted);
    font-size: 0.92rem;
    line-height: 1.45;
  }

  .selector-status {
    display: flex;
    align-items: flex-start;
    justify-content: space-between;
    gap: 1rem;
    margin-bottom: 0.9rem;
  }

  .selector-buffered {
    flex: 1;
    font-size: 0.95rem;
    line-height: 1.45;
    word-break: break-word;
  }

  .selector-count-wrap {
    min-width: 96px;
    text-align: right;
  }

  .selector-count-label {
    color: var(--muted);
    font-size: 0.78rem;
    margin-bottom: 0.15rem;
  }

  .selector-count-value {
    font-size: 2rem;
    line-height: 1;
  }

  .selector-actions {
    display: grid;
    grid-template-columns: minmax(0, 1fr) auto auto;
    gap: 0.75rem;
    align-items: end;
  }

  .selector-field {
    display: flex;
    flex-direction: column;
    gap: 0.35rem;
  }

  .selector-label {
    font-size: 0.8rem;
    color: var(--text);
  }

  .selector-select {
    width: 100%;
    min-height: 2.4rem;
    border: 1px solid var(--border);
    border-radius: 0.45rem;
    background: var(--bg);
    color: var(--text);
    padding: 0.5rem 0.75rem;
    font: inherit;
    appearance: auto;
    color-scheme: dark;
  }

  .selector-select:focus {
    outline: 1px solid var(--primary);
    outline-offset: 1px;
    border-color: var(--primary);
  }

  .selector-select option,
  .selector-select optgroup {
    background: var(--bg);
    color: var(--text);
  }

  .selector-button {
    min-height: 2.4rem;
    border-radius: 0.45rem;
    border: 1px solid var(--border);
    background: transparent;
    color: var(--text);
    padding: 0.5rem 0.9rem;
    font: inherit;
    cursor: pointer;
  }

  .selector-button-primary {
    border-color: rgba(255, 75, 75, 0.55);
    background: rgba(255, 75, 75, 0.18);
  }

  .selector-button:disabled {
    opacity: 0.45;
    cursor: not-allowed;
  }

  .selector-message {
    margin-top: 0.65rem;
    color: var(--muted);
    font-size: 0.82rem;
    min-height: 1.1rem;
  }

  .selector-marker-summary {
    margin: 0.45rem 0 0.95rem 0;
    color: var(--muted);
    font-size: 0.9rem;
    line-height: 1.5;
    word-break: break-word;
  }

  .selector-marker-summary strong {
    color: var(--text);
    font-weight: 600;
  }

  .js-plotly-plot .plotly .nsewdrag,
  .js-plotly-plot .plotly .drag,
  .js-plotly-plot .plotly .cursor-crosshair {
    cursor: pointer !important;
  }
`

const plotContainer = root.appendChild(document.createElement("div"))
plotContainer.id = "plotly-heatmap-container"

const panel = root.appendChild(document.createElement("div"))
panel.className = "selector-panel"

const caption = panel.appendChild(document.createElement("p"))
caption.className = "selector-caption"
caption.textContent = "Click any cell to toggle its cluster. Buffered clusters are framed in amber; clusters already assigned are framed in green. Clicks stay local until you apply."

const statusRow = panel.appendChild(document.createElement("div"))
statusRow.className = "selector-status"

const bufferedClusters = statusRow.appendChild(document.createElement("div"))
bufferedClusters.className = "selector-buffered"

const countWrap = statusRow.appendChild(document.createElement("div"))
countWrap.className = "selector-count-wrap"

const countLabel = countWrap.appendChild(document.createElement("div"))
countLabel.className = "selector-count-label"
countLabel.textContent = "Selected"

const countValue = countWrap.appendChild(document.createElement("div"))
countValue.className = "selector-count-value"

const markerSummary = panel.appendChild(document.createElement("div"))
markerSummary.className = "selector-marker-summary"

const actionsRow = panel.appendChild(document.createElement("div"))
actionsRow.className = "selector-actions"

const fieldWrap = actionsRow.appendChild(document.createElement("div"))
fieldWrap.className = "selector-field"

const selectLabel = fieldWrap.appendChild(document.createElement("label"))
selectLabel.className = "selector-label"
selectLabel.textContent = "Assign selected clusters to phenotype"

const phenotypeSelect = fieldWrap.appendChild(document.createElement("select"))
phenotypeSelect.className = "selector-select"

const applyButton = actionsRow.appendChild(document.createElement("button"))
applyButton.className = "selector-button selector-button-primary"
applyButton.textContent = "Assign Selected"

const clearButton = actionsRow.appendChild(document.createElement("button"))
clearButton.className = "selector-button"
clearButton.textContent = "Clear Selection"

const messageLine = panel.appendChild(document.createElement("div"))
messageLine.className = "selector-message"

let lastResetCounter = -1
let lastEventId = 0
let selectedClusters = new Set<string>()
let clusterDisplayLabels: Record<string, string> = {}
let clusterMarkerScores: Record<string, Record<string, number>> = {}
let baseLayoutShapes: any[] = []
let clusterOrder: string[] = []
let clusterIndexMap: Record<string, number> = {}
// Which axis the clusters are laid out on. "x" keeps the original column
// layout (markers on Y); "y" puts clusters on rows, matching the seaborn
// clustermap, in which case the highlight frames span rows horizontally.
let clusterAxis: "x" | "y" = "x"
// Length of the other axis, used to make a highlight frame span the full heatmap.
let crossAxisCount = 0
let plotlyPromise: Promise<any> | null = null

function setThemeStyles(theme?: RenderData["theme"]): void {
  const background = theme?.backgroundColor || "#0e1117"
  const textColor = theme?.textColor || "#fafafa"
  const primaryColor = theme?.primaryColor || "#ff4b4b"
  const fontFamily = theme?.font || "sans-serif"

  document.documentElement.style.setProperty("--bg", background)
  document.documentElement.style.setProperty("--text", textColor)
  document.documentElement.style.setProperty("--muted", `${textColor}B8`)
  document.documentElement.style.setProperty("--border", `${textColor}2E`)
  document.documentElement.style.setProperty("--surface", `${textColor}0A`)
  document.documentElement.style.setProperty("--primary", primaryColor)
  document.documentElement.style.setProperty("--font", fontFamily)
}

function ensurePlotly(): Promise<any> {
  if (window.Plotly) {
    return Promise.resolve(window.Plotly)
  }
  if (plotlyPromise) {
    return plotlyPromise
  }

  plotlyPromise = new Promise((resolve, reject) => {
    const script = document.createElement("script")
    script.src = plotlyUrl
    script.async = true
    script.onload = () => resolve(window.Plotly)
    script.onerror = () => reject(new Error("Failed to load Plotly"))
    document.head.appendChild(script)
  })

  return plotlyPromise
}

function normalizeClusterId(value: unknown): string | null {
  if (value === null || value === undefined) {
    return null
  }
  const normalized = String(value).trim()
  return normalized ? normalized : null
}

function getClusterDisplayLabel(clusterId: string): string {
  return clusterDisplayLabels[clusterId] || clusterId
}

function setMarkerSummaryText(text: string): void {
  markerSummary.textContent = text
}

function getTopMarkerSummary(selected: string[]): string {
  if (!selected.length) {
    return "Top 5 averaged markers: None"
  }

  const aggregate: Record<string, { sum: number; count: number }> = {}
  for (const clusterId of selected) {
    const scores = clusterMarkerScores[clusterId]
    if (!scores) {
      continue
    }
    for (const [marker, rawScore] of Object.entries(scores)) {
      const score = Number(rawScore)
      if (!Number.isFinite(score)) {
        continue
      }
      if (!aggregate[marker]) {
        aggregate[marker] = { sum: 0, count: 0 }
      }
      aggregate[marker].sum += score
      aggregate[marker].count += 1
    }
  }

  const topMarkers = Object.entries(aggregate)
    .map(([marker, stats]) => [marker, stats.sum / Math.max(1, stats.count)] as const)
    .sort((left, right) => right[1] - left[1])
    .slice(0, 5)

  if (!topMarkers.length) {
    return "Top 5 averaged markers: unavailable"
  }

  return `Top 5 averaged markers: ${topMarkers
    .map(([marker, score]) => `${marker} (${score.toFixed(2)})`)
    .join(", ")}`
}

function getAssignedClusterSet(): Set<string> {
  const assigned = new Set<string>()
  for (const shape of baseLayoutShapes) {
    const start = Number(clusterAxis === "y" ? shape?.y0 : shape?.x0)
    const end = Number(clusterAxis === "y" ? shape?.y1 : shape?.x1)
    if (!Number.isFinite(start) || !Number.isFinite(end)) {
      continue
    }
    const centerIndex = Math.round((start + end) / 2)
    const clusterId = clusterOrder[centerIndex]
    if (clusterId) {
      assigned.add(clusterId)
    }
  }
  return assigned
}

function buildSelectedClusterShapes(): any[] {
  if (!crossAxisCount || !clusterOrder.length) {
    return []
  }

  const assignedClusters = getAssignedClusterSet()
  return Array.from(selectedClusters)
    .filter((clusterId) => !assignedClusters.has(clusterId))
    .map((clusterId) => {
      const idx = clusterIndexMap[clusterId]
      if (!Number.isFinite(idx)) {
        return null
      }
      const band =
        clusterAxis === "y"
          ? { x0: -0.5, x1: crossAxisCount - 0.5, y0: idx - 0.5, y1: idx + 0.5 }
          : { x0: idx - 0.5, x1: idx + 0.5, y0: -0.5, y1: crossAxisCount - 0.5 }
      return {
        type: "rect",
        ...band,
        line: {
          color: "rgba(255, 193, 7, 0.98)",
          width: 3,
        },
        fillcolor: "rgba(255, 193, 7, 0.10)",
      }
    })
    .filter((shape) => shape !== null)
}

function syncClusterHighlightShapes(): void {
  if (!window.Plotly || !(plotContainer as any)?.layout) {
    return
  }
  const shapes = [...baseLayoutShapes, ...buildSelectedClusterShapes()]
  window.Plotly.relayout(plotContainer, { shapes })
}

function updateSelectionUi(): void {
  const selected = Array.from(selectedClusters)
  bufferedClusters.textContent = selected.length
    ? `Buffered clusters: ${selected.map(getClusterDisplayLabel).join(", ")}`
    : "Buffered clusters: None"
  setMarkerSummaryText(getTopMarkerSummary(selected))
  countValue.textContent = String(selected.length)
  applyButton.disabled = selected.length === 0 || !phenotypeSelect.value
  clearButton.disabled = selected.length === 0
  syncClusterHighlightShapes()
  Streamlit.setFrameHeight()
}

function rebuildPhenotypeOptions(options: string[]): void {
  const previousValue = phenotypeSelect.value
  phenotypeSelect.replaceChildren()

  for (const phenotype of options) {
    const option = document.createElement("option")
    option.value = phenotype
    option.textContent = phenotype
    phenotypeSelect.appendChild(option)
  }

  if (options.includes(previousValue)) {
    phenotypeSelect.value = previousValue
  } else if (options.length > 0) {
    phenotypeSelect.value = options[0]
  }
}

function deepClone<T>(value: T): T {
  return JSON.parse(JSON.stringify(value)) as T
}

function toggleClusterSelection(clusterId: string): void {
  if (selectedClusters.has(clusterId)) {
    selectedClusters.delete(clusterId)
  } else {
    selectedClusters.add(clusterId)
  }
  messageLine.textContent = ""
  updateSelectionUi()
}

function extractClusterId(point: any): string | null {
  if (!point) {
    return null
  }

  let clusterValue = point.customdata
  if (Array.isArray(clusterValue)) {
    clusterValue = clusterValue[0]
  }

  return normalizeClusterId(clusterValue)
}

async function renderComponent(data: RenderData): Promise<void> {
  setThemeStyles(data.theme)

  const availablePhenotypes = Array.isArray(data.args["availablePhenotypes"])
    ? data.args["availablePhenotypes"].map((value: unknown) => String(value))
    : []
  clusterDisplayLabels = Object.fromEntries(
    Object.entries((data.args["clusterDisplayLabels"] || {}) as Record<string, string>).map(
      ([clusterId, label]) => [String(clusterId), String(label)]
    )
  )
  clusterMarkerScores = Object.fromEntries(
    Object.entries((data.args["clusterMarkerScores"] || {}) as Record<string, Record<string, number>>).map(
      ([clusterId, markerScores]) => [
        String(clusterId),
        Object.fromEntries(
          Object.entries(markerScores || {}).map(([marker, score]) => [String(marker), Number(score)])
        ),
      ]
    )
  )

  rebuildPhenotypeOptions(availablePhenotypes)

  const resetCounter = Number(data.args["resetCounter"] || 0)
  if (resetCounter !== lastResetCounter) {
    selectedClusters = new Set<string>()
    lastResetCounter = resetCounter
    messageLine.textContent = ""
  }

  const figure = deepClone(data.args["figure"] || {})
  figure.layout = figure.layout || {}
  figure.layout.dragmode = false
  figure.layout.clickmode = "event"
  baseLayoutShapes = Array.isArray(figure.layout.shapes) ? deepClone(figure.layout.shapes) : []
  clusterAxis = data.args["clusterAxis"] === "y" ? "y" : "x"

  // customdata is a 2-D grid of cluster ids. With clusters on X every row
  // repeats the cluster order, so row 0 is the order; with clusters on Y every
  // column repeats it, so the first cell of each row is that row's cluster.
  const figureCustomdata = figure?.data?.[0]?.customdata
  const customdataRows: any[] = Array.isArray(figureCustomdata) ? figureCustomdata : []
  const rawClusterOrder =
    clusterAxis === "y"
      ? customdataRows.map((row: any) => (Array.isArray(row) ? row[0] : row))
      : Array.isArray(customdataRows[0])
      ? customdataRows[0]
      : []
  clusterOrder = rawClusterOrder
    .map((value: unknown) => normalizeClusterId(value))
    .filter((value: string | null): value is string => value !== null)
  clusterIndexMap = Object.fromEntries(clusterOrder.map((clusterId, index) => [clusterId, index]))
  const crossAxisValues = clusterAxis === "y" ? figure?.data?.[0]?.x : figure?.data?.[0]?.y
  crossAxisCount = Array.isArray(crossAxisValues) ? crossAxisValues.length : 0
  if (!figure.layout.paper_bgcolor && data.theme?.backgroundColor) {
    figure.layout.paper_bgcolor = data.theme.backgroundColor
  }
  if (!figure.layout.font && data.theme?.textColor) {
    figure.layout.font = {
      color: data.theme.textColor,
      family: data.theme.font,
    }
  } else if (figure.layout.font && !figure.layout.font.color && data.theme?.textColor) {
    figure.layout.font.color = data.theme.textColor
  }

  const plotlyConfig = {
    responsive: true,
    displayModeBar: false,
    scrollZoom: false,
    ...(figure.config || {}),
  }

  const Plotly = await ensurePlotly()
  await Plotly.react(plotContainer, figure.data || [], figure.layout, plotlyConfig)

  if (typeof (plotContainer as any).removeAllListeners === "function") {
    ;(plotContainer as any).removeAllListeners("plotly_click")
  }
  ;(plotContainer as any).on("plotly_click", (event: any) => {
    const clusterId = extractClusterId(event?.points?.[0])
    if (!clusterId) {
      return
    }
    toggleClusterSelection(clusterId)
  })

  updateSelectionUi()
  Streamlit.setFrameHeight()
}

phenotypeSelect.onchange = () => {
  updateSelectionUi()
}

applyButton.onclick = () => {
  const selected = Array.from(selectedClusters)
  if (!selected.length || !phenotypeSelect.value) {
    return
  }

  lastEventId += 1
  messageLine.textContent = `Applying '${phenotypeSelect.value}' to ${selected.length} cluster(s)...`
  Streamlit.setComponentValue({
    action: "apply",
    eventId: lastEventId,
    phenotype: phenotypeSelect.value,
    selectedClusters: selected,
  })
}

clearButton.onclick = () => {
  selectedClusters = new Set<string>()
  messageLine.textContent = "Selection cleared."
  updateSelectionUi()
}

function onRender(event: Event): void {
  renderComponent((event as CustomEvent<RenderData>).detail).catch((error: unknown) => {
    messageLine.textContent = error instanceof Error ? error.message : "Heatmap selector failed to render."
    Streamlit.setFrameHeight()
  })
}

Streamlit.events.addEventListener(Streamlit.RENDER_EVENT, onRender)
Streamlit.setComponentReady()
Streamlit.setFrameHeight()