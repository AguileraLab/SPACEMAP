// Cellpose correction geometry uses the polygon editor, but belongs to Cells and maps.
// Saved tiles stay a single SVG path; neither source is an annotation or a cell-data table.
function isTrainingCellLayer(layer) { return !!layer.cell_training; }
function trainingCellLayers() { return state.layers.filter(isTrainingCellLayer); }
function trainingCellStyle() {
  return state.trainingCellAppearance || (state.trainingCellAppearance = {mode:'outline', colour:'#00e5ff', opacity:1});
}
function trainingCellsAvailable() { return trainingCellLayers().length > 0 || state.trainingCells?.savedCount > 0; }
function trainingCellsVisible() {
  return trainingCellLayers().some(layer => layer.visible) || !!(state.trainingCells?.showSaved && state.trainingCells?.savedCount);
}

function trainingCellLayer(kind) {
  let layer = trainingCellLayers().find(layer => layer.cell_training === kind);
  if (!layer) {
    const image = state.layers.find(layer => layer.type === 'image');
    layer = {id:'training-cells-' + kind, type:'shapes', cell_training:kind, local:kind === 'current' ? 'Cells to correct' : 'Saved tile cells',
      name:kind === 'current' ? 'Current tile' : 'Saved cells', visible:true, opacity:trainingCellStyle().opacity,
      editable:kind === 'current', features:[], history:[], hidden:new Set(), info:image?.info || {},
      classes:[{name:'Cell', color:'#00e5ff'}]};
    state.layers.push(layer);
  }
  return layer;
}

function dropTrainingCellLayer(layer) {
  if (!layer) return;
  state.layers = state.layers.filter(item => item !== layer);
  if (state.drawInto === layer.id) state.drawInto = null;
  if (state.selected?.layer === layer) state.selected = null;
  if (state.picks?.layer === layer) state.picks = null;
  if (state.lastEdited === layer) state.lastEdited = null;
}

function editTrainingCells(name, polygons) {
  const layer = trainingCellLayer('current');
  layer.local = name;
  layer.features = polygons.map((ring, index) => ({type:'Feature',
    geometry:{type:'Polygon', coordinates:[ring.concat([ring[0]]).map(p => [Number(p[0]), Number(p[1])])]},
    properties:{id:newId(), name:'Cell ' + (index + 1), label:'Cell ' + (index + 1), classification:{name:'Cell', color:rgbOf('#00e5ff')}}}));
  layer.history = []; layer.future = []; layer.visible = true;
  state.drawInto = layer.id; state.selected = null; state.picks = null;
  (state.layerGroups || (state.layerGroups = {})).cells = true;
  renderShapes(); updateToolbar(); buildPanel();
  return layer.features.length;
}

function showSavedTrainingCells(outlines) {
  if (outlines) {
    const layer = trainingCellLayer('saved');
    layer.cellOutlines = outlines;
    layer.cellPath = outlines.map(ring => ring.map((p, i) => (i ? 'L' : 'M') + p[0] + ' ' + p[1]).join('') + 'Z').join('');
    layer.visible = true;
  } else dropTrainingCellLayer(trainingCellLayers().find(layer => layer.cell_training === 'saved'));
  renderShapes(); buildPanel();
}

function configureTrainingCells(controls) {
  state.trainingCells = controls;
  if (!controls) {
    trainingCellLayers().forEach(dropTrainingCellLayer);
    renderShapes(); updateToolbar();
  }
  buildPanel();
}

function styleTrainingCellPath(path, selected = false) {
  const style = trainingCellStyle();
  path.style.stroke = selected ? '#ffffff' : style.mode === 'filled' ? 'none' : style.colour;
  path.style.fill = style.colour;
  path.style.fillOpacity = style.mode === 'outline' ? '0' : selected ? '.35' : '.22';
}

function trainingCellsBlock() {
  const controls = state.trainingCells || {}, current = trainingCellLayers().find(layer => layer.cell_training === 'current');
  const saved = trainingCellLayers().find(layer => layer.cell_training === 'saved'), style = trainingCellStyle();
  const blocks = state.annotationBlocksOpen || (state.annotationBlocksOpen = {});
  const block = el('details', {class:'layer-group annotation-subgroup', 'data-cell-layer':'training', 'data-training-cells':'true'});
  block.open = blocks['training-cells'] ?? true;
  block.ontoggle = () => { if (block.isConnected) blocks['training-cells'] = block.open; };
  const all = el('input', {type:'checkbox', 'aria-label':'Show training cells'});
  const flags = [...(current ? [current.visible] : []), ...(controls.savedCount ? [!!controls.showSaved] : [])];
  all.checked = flags.some(Boolean); all.indeterminate = flags.some(Boolean) && !flags.every(Boolean);
  all.onclick = event => event.stopPropagation();
  all.onchange = () => {
    if (current) setVisible(current, all.checked);
    if (controls.savedCount) controls.onSaved?.(all.checked);
    buildPanel();
  };
  block.append(el('summary', {}, [all, el('span', {class:'name', text:'Training cells'}),
    help('Current tile: predicted cells you can correct and Keep tile. Saved cells: corrected training and validation cells on this image. These overlays are separate from the original segmentation and annotations.')]));
  const body = el('div', {class:'cell-settings'}); block.append(body);
  const source = (name, shown, count, action, disabled = false) => {
    const check = el('input', {type:'checkbox', 'aria-label':'Show ' + name.toLowerCase() + (name === 'Current tile' ? ' cells' : '')});
    check.disabled = disabled; check.checked = shown; check.onchange = () => action(check.checked);
    return el('div', {class:'row'}, [el('label', {class:'inline grow'}, [check, el('span', {text:name})]),
      el('span', {class:'muted', text:count})]);
  };
  if (current) body.append(source('Current tile', !!current.visible, current.features.length.toLocaleString() + ' cells', on => {setVisible(current, on); buildPanel();}));
  body.append(source('Saved cells', !!controls.showSaved,
    controls.loading ? 'Loading…' : (saved?.cellOutlines.length ?? controls.savedCount ?? 0).toLocaleString() + ' cells',
    on => controls.onSaved?.(on), !controls.savedCount));
  if (controls.error) body.append(el('div', {class:'hint', role:'status', text:controls.error}));
  if (saved && saved.cellOutlines.length < controls.savedCount) body.append(el('span', {class:'muted',
    text:saved.cellOutlines.length.toLocaleString() + ' of ' + controls.savedCount.toLocaleString() + ' cells shown'}));
  const redraw = () => {
    trainingCellLayers().forEach(layer => { layer.opacity = style.opacity; });
    renderShapes();
  };
  const mode = select([['outline','Outlines'], ['filled','Filled'], ['both','Filled + outlines']], style.mode, value => {style.mode = value; redraw();});
  mode.setAttribute('aria-label', 'Training cell appearance');
  const colour = el('input', {type:'color', value:style.colour, 'aria-label':'Training cell colour'});
  colour.oninput = () => {style.colour = colour.value; redraw();};
  const opacity = el('input', {type:'range', min:'0', max:'100', value:String(style.opacity * 100), 'aria-label':'Training cell opacity'});
  opacity.oninput = () => {style.opacity = Number(opacity.value) / 100; redraw();};
  body.append(cellControlRow('Appearance', [mode, colour, opacity]));
  if (controls.onTiles) {
    const tiles = el('input', {type:'checkbox', 'aria-label':'Show training tile outlines'}); tiles.checked = !!controls.showTiles;
    tiles.onchange = () => controls.onTiles(tiles.checked);
    body.append(el('div', {class:'row'}, [el('label', {class:'inline'}, [tiles, el('span', {text:'Tile outlines'})])]));
  }
  return block;
}
