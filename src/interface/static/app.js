"use strict";
const $ = (id) => document.getElementById(id);
const map = L.map("map", {preferCanvas: true}).setView([37.8716, -122.2727], 16);
const streets = L.tileLayer("https://tile.openstreetmap.org/{z}/{x}/{y}.png", {
  maxZoom: 20, maxNativeZoom: 19, attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap contributors</a>'
}).addTo(map);
let satellite = L.tileLayer("/api/imagery/tiles/{z}/{x}/{y}.png", {
  minZoom: 8, maxZoom: 20, attribution: "USGS / USDA NAIP · Geoportal Berlin / DOP 2025",
  updateWhenIdle: true, keepBuffer: 1
});
const layers = L.featureGroup().addTo(map);
const selectionLayer = L.featureGroup().addTo(map);
let settings = null, latestCollection = null, allCollection = null, currentResult = null;
let rectangle = null, selectedBounds = null, drawMode = false, drawStart = null, drawPointer = null;
let activeJob = null, trainingJob = null, analysedOverlay = null, imageryMode = false;
let currentPlace = "", previousQuery = "", modelSelection = "", pendingCombined = false;
let selectedFeature = null, selectedFeatureLayer = null, layerByFeature = new Map();
let beforeImage = null, afterImage = null, landmarkPairs = [], pendingLandmark = null, alignmentReady = false;
let selectionVersion = 0, selectionPlanTimer = null;

new ResizeObserver(() => map.invalidateSize({pan: false})).observe($("map"));
function setStatus(message, type = "") { $("status").textContent = message; $("status").className = type; }
async function request(url, options = {}) {
  const response = await fetch(url, options);
  let data;
  try { data = await response.json(); } catch { throw new Error(`Server returned HTTP ${response.status}`); }
  if (!response.ok) throw new Error(typeof data.detail === "string" ? data.detail : JSON.stringify(data.detail || data));
  return data;
}
const post = (url, data) => request(url, {method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify(data)});
async function busy(action, message, button = null) {
  if (button) button.disabled = true;
  setStatus(message, "busy");
  try { return await action(); } catch (error) { setStatus(error.message, "error"); }
  finally { if (button) button.disabled = false; }
}
function arrayBounds(b) { return [b.getWest(), b.getSouth(), b.getEast(), b.getNorth()]; }
function bounds() { return arrayBounds(selectedBounds || map.getBounds()); }
function featureLabel(p) { return p.reviewed_label || p.name || p.address || (p.candidate_name ? `${p.candidate_name} (candidate)` : null) || (p.candidate_address ? `${p.candidate_address} (candidate)` : null) || p.label || p.visual_class || p.change || "Mapped feature"; }
function hasAddress(p) { return Boolean(p.address || p.candidate_address || p.map_evidence?.some(e => e.address)); }
function areaKm2(b) { const [w,s,e,n] = arrayBounds(b); return Math.abs((e-w)*111.32*Math.cos((n+s)/2*Math.PI/180)*(n-s)*111.32); }
function refreshLocation() {
  const c = map.getCenter();
  $("map-location").textContent = `${c.lat.toFixed(5)}, ${c.lng.toFixed(5)} · zoom ${map.getZoom()}`;
}
map.on("moveend", refreshLocation); refreshLocation();

function setBasemap(useImagery) {
  imageryMode = useImagery;
  if (useImagery) { map.removeLayer(streets); if (!map.hasLayer(satellite)) satellite.addTo(map); }
  else { map.removeLayer(satellite); if (!map.hasLayer(streets)) streets.addTo(map); }
  $("street-map").classList.toggle("active", !useImagery);
  $("satellite-map").classList.toggle("active", useImagery);
  $("street-map").setAttribute("aria-pressed", String(!useImagery));
  $("satellite-map").setAttribute("aria-pressed", String(useImagery));
  if (!useImagery) $("imagery-source").textContent = "Street map · OpenStreetMap";
  else $("imagery-source").textContent = "Imagery · Berlin / contiguous US, or configured WMS";
}
$("street-map").onclick = () => setBasemap(false);
$("satellite-map").onclick = () => { setBasemap(true); if (map.getZoom() < 8) setStatus("Zoom in to see available aerial imagery."); };
satellite.on("tileerror", () => { if (imageryMode && !activeJob) setStatus("Some imagery tiles could not load. Retry or check the provider in Settings.", "error"); });

function exitDrawing() {
  drawMode = false; drawStart = null; drawPointer = null;
  map.dragging.enable(); map.doubleClickZoom.enable();
  $("map").classList.remove("drawing");
  $("draw-area").setAttribute("aria-pressed", "false");
  $("draw-area").textContent = "▭ Draw area";
}
function beginDrawing() {
  if (drawMode) { exitDrawing(); return; }
  activatePanel("search-panel"); setBasemap(true);
  drawMode = true; map.dragging.disable(); map.doubleClickZoom.disable();
  $("map").classList.add("drawing");
  $("draw-area").setAttribute("aria-pressed", "true"); $("draw-area").textContent = "Cancel drawing";
  setStatus("Drag a box over the area to analyse. Press Escape to cancel.");
}
$("draw-area").onclick = beginDrawing;
document.addEventListener("keydown", (event) => { if (event.key === "Escape") { exitDrawing(); rebuildSelection(); } });
function pointerLatLng(event) { const box = $("map").getBoundingClientRect(); return map.containerPointToLatLng([event.clientX-box.left,event.clientY-box.top]); }
$("map").addEventListener("pointerdown", (event) => {
  if (!drawMode || event.button !== 0 || event.target.closest(".leaflet-control")) return;
  event.preventDefault(); drawPointer = event.pointerId; drawStart = pointerLatLng(event);
  $("map").setPointerCapture(event.pointerId); selectionLayer.clearLayers();
  rectangle = L.rectangle(L.latLngBounds(drawStart, drawStart), {color: "#f2a53a", weight: 2, fillOpacity: .08, interactive: false}).addTo(selectionLayer);
});
$("map").addEventListener("pointermove", (event) => {
  if (!drawStart || event.pointerId !== drawPointer) return;
  rectangle.setBounds(L.latLngBounds(drawStart, pointerLatLng(event)));
});
$("map").addEventListener("pointerup", (event) => {
  if (!drawStart || event.pointerId !== drawPointer) return;
  const next = L.latLngBounds(drawStart, pointerLatLng(event));
  if ($("map").hasPointerCapture(event.pointerId)) $("map").releasePointerCapture(event.pointerId);
  exitDrawing();
  if (areaKm2(next) < .0001) { rebuildSelection(); setStatus("Drag a larger box to select an area."); return; }
  setSelection(next, true);
});
$("map").addEventListener("pointercancel", () => { exitDrawing(); rebuildSelection(); });
function rebuildSelection() {
  selectionLayer.clearLayers(); rectangle = null;
  if (!selectedBounds) return;
  rectangle = L.rectangle(selectedBounds, {color: "#f2a53a", weight: 2, fillOpacity: .035, interactive: false}).addTo(selectionLayer);
  const corners = [selectedBounds.getNorthWest(),selectedBounds.getNorthEast(),selectedBounds.getSouthEast(),selectedBounds.getSouthWest()];
  corners.forEach((corner, index) => {
    const opposite = corners[(index+2)%4];
    const marker = L.marker(corner, {draggable: true, icon: L.divIcon({className: "selection-handle", iconSize: [12,12], iconAnchor:[6,6]}), title:"Resize selected area"}).addTo(selectionLayer);
    marker.on("drag", () => rectangle.setBounds(L.latLngBounds(marker.getLatLng(), opposite)));
    marker.on("dragend", () => setSelection(rectangle.getBounds(), false));
  });
  const centre = selectedBounds.getCenter(), original = arrayBounds(selectedBounds);
  const handle = L.marker(centre, {draggable:true, icon:L.divIcon({className:"selection-move",html:"✥",iconSize:[24,24],iconAnchor:[12,12]}),title:"Move selected area"}).addTo(selectionLayer);
  handle.on("drag", () => { const delta = handle.getLatLng(); rectangle.setBounds([[original[1]+delta.lat-centre.lat,original[0]+delta.lng-centre.lng],[original[3]+delta.lat-centre.lat,original[2]+delta.lng-centre.lng]]); });
  handle.on("dragend", () => setSelection(rectangle.getBounds(), false));
}
async function inspectSelection(version) {
  if (!selectedBounds) return null;
  const plan = await post("/api/imagery/plan", {bounds: arrayBounds(selectedBounds)});
  if (version !== selectionVersion) return null;
  $("selection-info").textContent = `${areaKm2(selectedBounds).toFixed(3)} km² · ${plan.width} × ${plan.height} pixels · ${plan.inference_tiles} windows · ${plan.resolution_m} m/pixel`;
  if (imageryMode) $("imagery-source").textContent = `${plan.source.name} · ${plan.source.acquisition}`;
  return plan;
}
function setSelection(b, autoRun = false) {
  selectedBounds = b; selectionVersion++; rebuildSelection();
  $("selection-info").textContent = `${areaKm2(b).toFixed(3)} km² selected. Checking imagery…`;
  clearTimeout(selectionPlanTimer);
  const version = selectionVersion;
  selectionPlanTimer = setTimeout(async () => {
    try {
      const plan = await inspectSelection(version);
      if (plan && autoRun && $("auto-analyse").checked && !activeJob) await analyseArea();
      else if (plan && !activeJob) setStatus("Area selected. Choose a concept and click Analyse selected area.");
    } catch (error) { if (version === selectionVersion) { $("selection-info").textContent = error.message; setStatus(error.message, "error"); } }
  }, 180);
}
$("use-view").onclick = () => { exitDrawing(); setBasemap(true); setSelection(map.getBounds(), true); };
$("clear-selection").onclick = () => { exitDrawing(); clearTimeout(selectionPlanTimer); selectedBounds = null; selectionVersion++; rebuildSelection(); $("selection-info").textContent = "Draw a box, or use the visible area."; setStatus("Selection cleared."); };

async function followJob(identifier) {
  if (activeJob && activeJob !== identifier) throw new Error("An analysis is already running.");
  activeJob = identifier; sessionStorage.setItem("geo-active-job", identifier);
  $("job-progress").hidden = false; $("analyse-area").disabled = true;
  try {
    let failures = 0;
    while (true) {
      let data;
      try { data = await request(`/api/jobs/${identifier}`); failures = 0; }
      catch (error) { if (++failures >= 3) throw error; await new Promise(r => setTimeout(r, 1500)); continue; }
      $("progress-meter").value = data.progress; setStatus(data.message, data.status === "failed" ? "error" : "busy");
      if (data.status === "completed") return data.result;
      if (["failed", "cancelled"].includes(data.status)) throw new Error(data.message);
      await new Promise(resolve => setTimeout(resolve, 1200));
    }
  } finally { activeJob = null; sessionStorage.removeItem("geo-active-job"); $("job-progress").hidden = true; $("analyse-area").disabled = false; }
}
$("cancel-analysis").onclick = () => { if (activeJob) busy(async () => { await request(`/api/jobs/${activeJob}/cancel`, {method:"POST"}); setStatus("Cancellation requested; the current operation may need to finish."); }, "Cancelling…"); };
async function analyseArea(prompt = null, enrich = null, sites = null) {
  if (activeJob) { setStatus("Analysis is running. Wait or cancel it before starting another area."); return; }
  return busy(async () => {
    if (!settings) throw new Error("Settings are still loading. Try again in a moment.");
    const concept = (prompt || $("concept").value).trim();
    if (!concept) throw new Error("Enter a visible concept, such as building or tree.");
    $("concept").value = concept; setBasemap(true);
    const extent = bounds();
    const plan = await post("/api/imagery/plan", {bounds: extent});
    $("imagery-source").textContent = `${plan.source.name} · ${plan.source.acquisition}`;
    const started = await post("/api/segment-view", {bounds: extent, prompt: concept, enrich: enrich ?? settings.map.enrich,
      site_category: sites?.category || null, site_tag: sites?.tag || null});
    const result = await followJob(started.job_id); renderImages(result);
  }, "Checking imagery and selected area…", $("analyse-area"));
}
$("analyse-area").onclick = () => analyseArea();
document.querySelectorAll("[data-concept]").forEach(button => button.onclick = () => { $("concept").value = button.dataset.concept; });

function filtered(collection) {
  const filter = $("result-filter").value;
  return {...collection, features: collection.features.filter(f => {
    const p = f.properties || {};
    if (filter === "rejected") return p.rejected;
    if (p.rejected) return false;
    return filter === "all" || (filter === "has_address" ? hasAddress(p) : !hasAddress(p));
  })};
}
function drawResults() {
  layers.clearLayers(); layerByFeature.clear(); $("result-list").replaceChildren();
  if (!allCollection) return;
  latestCollection = filtered(allCollection);
  const colors = {building:"#20d2b0",tree:"#77cf68",water:"#48a9ff",road:"#ffca62"};
  const vector = L.geoJSON(latestCollection, {
    style: f => ({color:f.properties?.change?.startsWith("removed") ? "#ff5874" : colors[f.properties?.visual_class] || "#20bfa7",weight:2,fillOpacity:.16}),
    pointToLayer: (f,ll) => L.circleMarker(ll,{radius:7,color:"#fff",weight:2,fillColor:"#087e70",fillOpacity:1}),
    onEachFeature: (feature, layer) => {
      layerByFeature.set(feature, layer);
      const label = document.createElement("span"); label.textContent = featureLabel(feature.properties || {});
      layer.bindTooltip(label, {permanent: $("show-labels").checked && latestCollection.features.length <= 100, direction:"center",className:"feature-label"});
      layer.on("click", () => showFeature(feature, layer));
      const button = document.createElement("button"); button.className = "result-item";
      const title = document.createElement("strong"); title.textContent = label.textContent;
      const detail = document.createElement("span"), p = feature.properties || {};
      detail.textContent = [p.visual_class, p.score !== undefined ? `score ${p.score.toFixed(2)}` : null, hasAddress(p) ? "Address evidence" : "Address unknown"].filter(Boolean).join(" · ");
      button.append(title,detail); button.onclick = () => { map.fitBounds(layer.getBounds ? layer.getBounds() : L.latLngBounds(layer.getLatLng(),layer.getLatLng()),{maxZoom:19,padding:[45,45]}); showFeature(feature,layer); };
      $("result-list").append(button);
    }
  });
  layers.addLayer(vector);
  if (!$("show-results").checked) map.removeLayer(layers); else if (!map.hasLayer(layers)) layers.addTo(map);
  $("result-count").textContent = latestCollection.features.length;
  $("result-summary").textContent = `${latestCollection.features.length} features · ${latestCollection.attribution || "User data"}`;
  $("download").disabled = !latestCollection.features.length;
  $("download-csv").disabled = !latestCollection.features.length;
  $("enrich-results").disabled = !allCollection.features.some(f => f.properties?.visual_class);
  if (!latestCollection.features.length) $("result-list").textContent = "No results match this filter.";
}
function render(collection, {fit=true, preserveImage=false} = {}) {
  if (!preserveImage) {
    if (analysedOverlay) map.removeLayer(analysedOverlay);
    analysedOverlay = null; currentResult = null; $("image-results").hidden = true;
    $("download-image").hidden = true; $("download-pixels").hidden = true;
  }
  allCollection = collection; selectedFeature = null; $("feature-details").hidden = true;
  pendingCombined = false;
  $("result-filter").value = "all"; drawResults();
  if (fit && layers.getBounds().isValid()) map.fitBounds(layers.getBounds(), {padding:[40,40],maxZoom:18});
}
function showFeature(feature, layer) {
  selectedFeature = feature; selectedFeatureLayer = layer;
  const root = $("feature-details"); root.hidden = false; root.replaceChildren();
  const p = feature.properties || {}, top = document.createElement("div"); top.className = "viewer-heading";
  const title = document.createElement("h2"); title.textContent = featureLabel(p);
  const close = document.createElement("button"); close.textContent = "Close"; close.className = "secondary"; close.onclick = () => root.hidden = true;
  top.append(title,close); root.append(top);
  const text = document.createElement("p"); text.className = "hint";
  text.textContent = [p.visual_class, p.score !== undefined ? `Model score ${p.score.toFixed(3)}` : null, p.address ? `Mapped address: ${p.address}` : p.candidate_address ? `Candidate address: ${p.candidate_address}` : "Address not established", p.match_status].filter(Boolean).join(" · "); root.append(text);
  if (p.visual_class && p.object_id) {
    const form = document.createElement("form"); form.className = "label-editor";
    const label = document.createElement("label"); label.textContent = "Your reviewed label";
    const input = document.createElement("input"); input.value = p.reviewed_label || p.label || p.visual_class; input.maxLength=200; label.append(input);
    const rejected = document.createElement("label"); rejected.className="check";
    const check = document.createElement("input"); check.type="checkbox"; check.checked=Boolean(p.rejected); rejected.append(check,document.createTextNode("Reject detection"));
    const save = document.createElement("button"); save.textContent="Save label"; save.type="submit";
    form.append(label,rejected,save); form.onsubmit = event => { event.preventDefault(); busy(async () => {
      const edit = {object_id:p.object_id,reviewed_label:input.value.trim(),rejected:check.checked};
      if (currentResult?.run_id) await request(`/api/results/${currentResult.run_id}/labels`,{method:"PATCH",headers:{"Content-Type":"application/json"},body:JSON.stringify([edit])});
      Object.assign(p,edit); drawResults(); showFeature(feature,layer);
      if (currentResult?.annotated_preview) $("download-image").href=currentResult.annotated_preview+`?v=${Date.now()}`;
      if (currentResult?.run_id) $("download-pixels").href=`/results/${currentResult.run_id}/image_features.json?v=${Date.now()}`;
      setStatus("Reviewed label saved. Exports include your edits.");
    },"Saving label…",save); }; root.append(form);
  }
  const details = document.createElement("details"), summary = document.createElement("summary"), pre = document.createElement("pre");
  summary.textContent="Map evidence and source properties"; pre.textContent=JSON.stringify(p,null,2); details.append(summary,pre); root.append(details);
  if (p.source_url && /^https:\/\/www\.openstreetmap\.org\//.test(p.source_url)) {
    const link = document.createElement("a"); link.href=p.source_url; link.textContent="Open source map record"; link.target="_blank"; link.rel="noopener"; root.append(link);
  }
}
$("result-filter").onchange = drawResults;
$("show-labels").onchange = drawResults;
$("show-results").onchange = () => { if ($("show-results").checked) layers.addTo(map); else map.removeLayer(layers); };
$("show-analysed").onchange = () => { if (analysedOverlay) { if ($("show-analysed").checked) analysedOverlay.addTo(map); else map.removeLayer(analysedOverlay); } };
async function enrichExisting() {
  if (!allCollection?.features.length) throw new Error("Run SAM in a selected area first.");
  const geoBounds = L.geoJSON(allCollection).getBounds();
  const result = await post("/api/enrich",{collection:allCollection,bounds:arrayBounds(geoBounds)});
  render(result,{fit:false,preserveImage:true}); setStatus("Map evidence attached. Spatial matches remain candidates until reviewed.");
}
$("enrich-results").onclick = () => busy(enrichExisting,"Matching names and addresses…",$("enrich-results"));

function showCandidates(data) {
  $("candidates").replaceChildren();
  if (!["candidates","area_candidates","segment_candidates"].includes(data.kind)) return;
  data.geojson.features.forEach(feature => {
    const p = feature.properties || {}, button = document.createElement("button");
    button.textContent = [p.name,p.address].filter(Boolean).join(" · ") || "Select this location";
    button.onclick = () => busy(async () => {
      currentPlace = button.textContent; selectedBounds=null; selectionVersion++; rebuildSelection();
      const [lon,lat] = feature.geometry.coordinates; map.setView([lat,lon],16);
      if (data.kind === "segment_candidates") {
        $("concept").value = data.plan.prompt || "building"; setBasemap(true); beginDrawing();
      } else if (data.kind === "area_candidates") {
        if (data.plan.category) $("category").value=data.plan.category;
        $("custom-tag").value=data.plan.tag || "";
        pendingCombined = data.plan.source === "combined";
        setStatus("Location selected. Draw a small area or use Search selected / visible area for map records.");
        document.querySelector(".map-records").open=true;
      } else if (p.osm_type && p.osm_id) {
        render(await request(`/api/resolve/${encodeURIComponent(p.osm_type)}/${p.osm_id}`));
        if (["imagery","combined"].includes($("search-source").value)) setBasemap(true);
        setStatus("Selected source feature highlighted. Draw an area here to run SAM.");
      } else { render({type:"FeatureCollection",features:[feature]}); setStatus("Location selected. Draw a box to analyse imagery here."); }
    },"Selecting location…",button);
    $("candidates").append(button);
  });
}
$("search-form").onsubmit = event => {
  event.preventDefault(); busy(async () => {
    const result = await post("/api/search",{query:$("query").value,bounds:bounds(),mode:$("search-mode").value,
      source:$("search-source").value,use_ollama:$("use-ollama").checked,
      context:{has_selection:Boolean(selectedBounds),bounds:bounds(),place:currentPlace,previous_query:previousQuery,
        result_count:allCollection?.features.length || 0,concept:$("concept").value}});
    previousQuery=$("query").value;
    if (result.kind === "segment") { await analyseArea(result.plan.prompt,result.plan.source === "combined" ? true : null); return; }
    if (result.kind === "enrich") { await enrichExisting(); return; }
    if (result.kind === "filter") { if (!allCollection) throw new Error("Search or run SAM first, then filter its results."); $("result-filter").value=result.plan.filter; drawResults(); setStatus("Filtered the existing result set."); return; }
    if (result.kind === "compare") { activatePanel("compare-panel"); setStatus(result.message); return; }
    render(result.geojson); showCandidates(result); setStatus(result.message);
    if (result.kind === "features" && result.plan.source === "combined") {
      await analyseArea("building",true,{category:result.plan.category,tag:result.plan.tag});
    }
  },"Understanding and locating your request…",event.submitter);
};
$("area-search").onclick = () => busy(async () => {
  if (pendingCombined) { await analyseArea("building",true,{category:$("category").value,tag:$("custom-tag").value.trim() || null}); return; }
  const result=await post("/api/features",{bounds:bounds(),category:$("category").value,tag:$("custom-tag").value.trim() || null});
  render(result,{fit:false}); if (pendingCombined) setBasemap(true);
  setStatus(result.features.length ? "Mapped records loaded. Click a result to inspect its name, address and source." : "No matching map records were returned.");
},"Fetching map records…",$("area-search"));
document.querySelectorAll("[data-query]").forEach(button => button.onclick=()=>{$("query").value=button.dataset.query;$("search-form").requestSubmit();});

function activatePanel(id, updateHistory=true) {
  document.querySelectorAll(".panel").forEach(panel=>panel.hidden=panel.id!==id);
  document.querySelectorAll("[data-panel]").forEach(button=>button.classList.toggle("active",button.dataset.panel===id));
  document.querySelector("main").classList.toggle("settings-open",id==="settings-panel");
  if (updateHistory) history.pushState({panel:id},"",id==="settings-panel"?"/settings":"/");
  map.invalidateSize();
}
document.querySelectorAll("[data-panel]").forEach(button=>button.onclick=()=>activatePanel(button.dataset.panel));
window.addEventListener("popstate",event=>activatePanel(event.state?.panel || (location.pathname==="/settings"?"settings-panel":"search-panel"),false));

function addPreview(url, caption) {
  const figure=document.createElement("figure"),img=document.createElement("img"),text=document.createElement("figcaption");
  img.src=url;img.alt=caption;text.textContent=caption;figure.append(img,text);$("previews").append(figure);
}
function renderImages(result) {
  if (result.kind==="model") {setStatus(result.message);return;}
  render(result.geojson,{fit:false}); currentResult=result;
  $("image-results").hidden=false;$("previews").replaceChildren();
  addPreview(result.preview,result.before_preview?"Change candidates · green added, red removed":"SAM detections across the analysed area");
  $("download-image").href=result.annotated_preview || result.preview;$("download-image").hidden=false;
  if (result.image_features) {$("download-pixels").href=`/results/${result.run_id}/image_features.json`;$("download-pixels").hidden=false;}
  if (result.map_image && result.map_bounds) {
    analysedOverlay=L.imageOverlay(result.map_image,result.map_bounds,{opacity:1,interactive:false});
    if ($("show-analysed").checked) analysedOverlay.addTo(map).bringToBack();
    if (!selectedBounds) map.fitBounds(result.map_bounds,{padding:[25,25]});
  }
  if (result.before_preview) showSwipe(result.before_preview,result.after_preview);
  const m=result.metadata;
  if (m.source) $("imagery-source").textContent=`${m.source.name} · ${m.source.acquisition}`;
  const info=document.createElement("p");
  info.textContent=m.added_pixels!==undefined?`Added: ${m.added_pixels} pixels · removed: ${m.removed_pixels} pixels · ${m.alignment?.method || ""}`:`${m.objects} detections · ${m.width || m.window?.[2]} × ${m.height || m.window?.[3]} pixels${m.georeferenced?" · map coordinates retained":" · image coordinates"}`;
  $("result-summary").append(info);
  setStatus(m.warnings?.length?m.warnings.join(" "):m.interpretation || `Finished. ${m.objects} detections. ${m.georeferenced?"Outlines placed on the map.":"Inspect the image preview and download image labels."}`);
}

function settingsValues() {
  const values=structuredClone(settings || {});
  delete values.imagery?.has_custom_token;
  document.querySelectorAll("[data-setting]").forEach(input=>{
    const [section,key]=input.dataset.setting.split("."); values[section] ||= {};
    values[section][key]=input.type==="checkbox"?input.checked:input.type==="number"?Number(input.value):input.value;
  });
  if (!values.ollama.model) values.ollama.model=null;
  if ($("clear-token").checked) values.imagery.clear_custom_token=true;
  return values;
}
function applySettings(data, initial=false) {
  settings=data;modelSelection=data.ollama.model || "";
  if (modelSelection && !Array.from($("ollama-model").options).some(o=>o.value===modelSelection)) {
    const option=new Option(`${modelSelection} (saved)`,modelSelection);$("ollama-model").add(option);
  }
  document.querySelectorAll("[data-setting]").forEach(input=>{
    const [section,key]=input.dataset.setting.split(".");const value=data[section]?.[key];
    if (input.type==="checkbox") input.checked=Boolean(value);else input.value=value ?? "";
  });
  $("clear-token").checked=false;
  $("concept").value=data.sam3.prompt;$("compare-concept").value=data.sam3.prompt;
  $("alignment").value=data.comparison.alignment; updateAlignmentMode();
  $("auto-analyse").checked=data.map.auto_analyse;$("show-labels").checked=data.map.show_labels;
  $("use-ollama").checked=data.ollama.enabled && Boolean(data.ollama.model);
  $("use-ollama").disabled=!data.ollama.model;
  $("ollama-hint").textContent=data.ollama.model?`Model: ${data.ollama.model}`:"Choose and test a model in Settings to enable broader language requests.";
  $("model-status").textContent=data.ollama.model?`Ollama: ${data.ollama.model}`:"Ollama: no model selected";
  $("custom-provider-fields").hidden=data.imagery.provider!=="custom_wms";
  if (initial) map.setView([data.map.default_lat,data.map.default_lon],data.map.default_zoom);
  satellite.setUrl(`/api/imagery/tiles/{z}/{x}/{y}.png?v=${Date.now()}`);
  if (allCollection) drawResults();
}
async function refreshModels() {
  const values=settingsValues().ollama; const selected=$("ollama-model").value || modelSelection;
  const data=await post("/api/ollama/models",values);
  $("ollama-connection").textContent=data.message;
  $("ollama-model").replaceChildren(new Option("Choose an installed model",""));
  data.models.forEach(model=>{const option=new Option(`${model.name} · ${(model.size/1024**3).toFixed(1)} GB${model.note?" · "+model.note:""}`,model.name);option.disabled=model.selectable===false;$("ollama-model").add(option);});
  if (selected && !data.models.some(m=>m.name===selected)) $("ollama-model").add(new Option(`${selected} (saved; unavailable)`,selected));
  $("ollama-model").value=selected || "";
  $("model-status").textContent=data.connected?`Ollama: ${settings.ollama.model || "choose a model"}`:"Ollama: offline";
  return data;
}
$("refresh-models").onclick=()=>busy(async()=>{const result=await refreshModels();setStatus(result.message,result.connected?"":"error");},"Checking Ollama…",$("refresh-models"));
$("test-model").onclick=()=>busy(async()=>{
  const data=await post("/api/ollama/test",{settings:settingsValues().ollama,query:$("model-test-query").value});
  $("model-test-result").hidden=false;$("model-test-result").textContent=JSON.stringify(data.plan,null,2);setStatus(data.message);
},"Testing model with a geographic request…",$("test-model"));
$("imagery-provider").onchange=()=>$("custom-provider-fields").hidden=$("imagery-provider").value!=="custom_wms";
$("settings-form").onsubmit=event=>{event.preventDefault();busy(async()=>{
  const data=await request("/api/settings",{method:"PUT",headers:{"Content-Type":"application/json"},body:JSON.stringify(settingsValues())});
  applySettings(data);$("settings-message").textContent="Saved. New analyses use these settings.";setStatus("Settings saved.");
},"Saving settings…",event.submitter);};
$("discard-settings").onclick=()=>{applySettings(settings);$("settings-message").textContent="Unsaved edits discarded.";};
$("inspect-sam").onclick=()=>busy(async()=>{const data=await request("/api/sam/status");$("sam-state").textContent=`${data.complete?"Checkpoint complete":"Missing: "+data.missing.join(", ")} · ${data.loaded?"loaded":"not loaded"} · ${data.snapshot}`;setStatus("Checkpoint inspected.");},"Inspecting local SAM files…",$("inspect-sam"));
$("reload-sam").onclick=()=>busy(async()=>{const start=await request("/api/sam/reload",{method:"POST"});const data=await followJob(start.job_id);$("sam-state").textContent=data.message;setStatus(data.message);},"Loading saved SAM settings…",$("reload-sam"));

function resetAlignment() {alignmentReady=false;$("compare-submit").disabled=true;$("alignment-result").textContent="Preview alignment before comparing.";}
function updateAlignmentMode() {const manual=$("alignment").value==="manual";$("manual-hint").hidden=!manual;$("clear-points").hidden=!manual;$("pair-previews").classList.toggle("marking",manual);if(manual && beforeImage && afterImage){$("compare-viewer").hidden=false;$("pair-previews").hidden=false;$("swipe-view").hidden=true;drawLandmarks();}resetAlignment();}
$("alignment").onchange=updateAlignmentMode;
function drawLandmarks() {
  [["before-canvas",beforeImage,0],["after-canvas",afterImage,1]].forEach(([id,img,side])=>{
    if (!img) return;const canvas=$(id),scale=Math.min(1,900/img.width);canvas.width=Math.round(img.width*scale);canvas.height=Math.round(img.height*scale);
    const ctx=canvas.getContext("2d");ctx.drawImage(img,0,0,canvas.width,canvas.height);
    const points=landmarkPairs.map(pair=>pair[side]);if (side===0 && pendingLandmark) points.push(pendingLandmark);
    points.forEach((p,i)=>{const x=p[0]*canvas.width,y=p[1]*canvas.height;ctx.fillStyle="#fcb94b";ctx.strokeStyle="#122b39";ctx.lineWidth=2;ctx.beginPath();ctx.arc(x,y,7,0,Math.PI*2);ctx.fill();ctx.stroke();ctx.font="bold 16px sans-serif";ctx.fillText(String(i+1),x+10,y-8);});
  });
  $("point-count").textContent=landmarkPairs.length?`${landmarkPairs.length} point pairs${pendingLandmark?" · click matching After landmark":""}`:pendingLandmark?"Click the matching landmark in After":"";
}
async function loadPairImage(input, side) {
  const file=input.files[0];if (!file) return;
  landmarkPairs=[];pendingLandmark=null;resetAlignment();
  if (/\.tiff?$/i.test(file.name)) {if (side===0) beforeImage=null;else afterImage=null;setStatus("GeoTIFF selected. Preview alignment to inspect the common raster grid.");return;}
  const url=URL.createObjectURL(file),img=new Image();
  try {await new Promise((resolve,reject)=>{img.onload=resolve;img.onerror=()=>reject(new Error("Cannot preview this image."));img.src=url;});if(side===0)beforeImage=img;else afterImage=img;}
  finally {URL.revokeObjectURL(url);}
  $("compare-viewer").hidden=false;$("pair-previews").hidden=false;$("swipe-view").hidden=true;$("viewer-title").textContent="Review your images";drawLandmarks();
  setStatus("Image loaded. Add both images, then preview their alignment.");
}
$("before-file").onchange=()=>busy(()=>loadPairImage($("before-file"),0),"Reading before image…");
$("after-file").onchange=()=>busy(()=>loadPairImage($("after-file"),1),"Reading after image…");
[["before-canvas",0],["after-canvas",1]].forEach(([id,side])=>$(id).onclick=event=>{
  if ($("alignment").value!=="manual" || !beforeImage || !afterImage) return;
  const rect=event.currentTarget.getBoundingClientRect(),point=[(event.clientX-rect.left)/rect.width,(event.clientY-rect.top)/rect.height];
  if(side===0) pendingLandmark=point;
  else if(pendingLandmark){landmarkPairs.push([pendingLandmark,point]);pendingLandmark=null;}
  else {setStatus("Click a Before landmark first, then its matching After landmark.");return;}
  resetAlignment();drawLandmarks();
});
$("clear-points").onclick=()=>{landmarkPairs=[];pendingLandmark=null;resetAlignment();drawLandmarks();};
$("compare-use-bounds").onclick=()=>{if(!selectedBounds){setStatus("Draw the exact extent of the north-up After image first.","error");return;}$("compare-bounds").value=arrayBounds(selectedBounds).join(",");resetAlignment();};
$("compare-form").addEventListener("change",resetAlignment);
function compareData() {const data=new FormData($("compare-form"));data.set("points",JSON.stringify(landmarkPairs));data.set("background","true");return data;}
function showSwipe(before,after) {$("compare-viewer").hidden=false;$("swipe-view").hidden=false;$("pair-previews").hidden=true;$("swipe-before").src=before;$("swipe-after").src=after;$("viewer-title").textContent="Aligned before / after";$("swipe-slider").dispatchEvent(new Event("input"));}
$("swipe-slider").oninput=()=>$("swipe-before").style.clipPath=`inset(0 ${100-Number($("swipe-slider").value)}% 0 0)`;
$("close-viewer").onclick=()=>$("compare-viewer").hidden=true;
$("preview-alignment").onclick=()=>{
  if(!$("compare-form").reportValidity())return;
  if(activeJob){setStatus("Wait for the current analysis first.");return;}
  busy(async()=>{
    const start=await request("/api/compare/align",{method:"POST",body:compareData()});const data=await followJob(start.job_id);
    showSwipe(data.before_preview,data.after_preview);$("previews").replaceChildren();$("image-results").hidden=false;addPreview(data.preview,"Alignment blend · check that stable objects overlap");
    alignmentReady=true;$("compare-submit").disabled=false;
    $("alignment-result").textContent=`${data.metadata.alignment.method} · ${(data.metadata.overlap_fraction*100).toFixed(1)}% overlap${data.metadata.alignment.registration_error_px!==undefined?` · ${data.metadata.alignment.registration_error_px} px landmark error`:""}. Review the preview, then compare.`;
    setStatus("Alignment ready. Inspect the slider and blended preview before running comparison.");
  },"Preparing alignment preview…",$("preview-alignment"));
};
$("compare-form").onsubmit=event=>{event.preventDefault();if(!alignmentReady){setStatus("Preview the alignment first.","error");return;}if(activeJob){setStatus("Wait for the current analysis first.");return;}busy(async()=>{const start=await request("/api/compare",{method:"POST",body:compareData()});renderImages(await followJob(start.job_id));},"Comparing aligned images…",$("compare-submit"));};

$("segment-form").onsubmit=event=>{event.preventDefault();if(activeJob)return;busy(async()=>{const data=new FormData(event.target);data.set("enrich",String(event.target.elements.enrich.checked));data.set("background","true");const start=await request("/api/segment",{method:"POST",body:data});renderImages(await followJob(start.job_id));},"Processing uploaded image…",event.submitter);};
$("demo").onclick=()=>busy(async()=>renderImages(await request("/api/demo",{method:"POST"})),"Processing cached Berkeley imagery…",$("demo"));
$("clear").onclick=()=>{layers.clearLayers();if(analysedOverlay)map.removeLayer(analysedOverlay);analysedOverlay=null;currentResult=null;allCollection=null;latestCollection=null;$("feature-details").hidden=true;$("image-results").hidden=true;$("candidates").replaceChildren();$("result-list").replaceChildren();$("result-summary").textContent="";$("result-count").textContent="0";$("download").disabled=true;$("download-csv").disabled=true;$("download-image").hidden=true;$("download-pixels").hidden=true;$("enrich-results").disabled=true;setStatus("Results cleared. The selected area is retained.");};
function downloadBlob(value,type,name){const url=URL.createObjectURL(new Blob([value],{type})),link=document.createElement("a");link.href=url;link.download=name;link.click();setTimeout(()=>URL.revokeObjectURL(url),1000);}
$("download").onclick=()=>{if(latestCollection)downloadBlob(JSON.stringify(latestCollection,null,2),"application/geo+json","geo_features.geojson");};
$("download-csv").onclick=()=>{
  if(!latestCollection)return;
  const columns=["object_id","label","visual_class","score","address","candidate_address","match_status","source"];
  const cell=value=>'"'+String(value??"").replaceAll('"','""')+'"';
  const rows=latestCollection.features.map(f=>columns.map(k=>cell(k==="label"?featureLabel(f.properties || {}):f.properties?.[k])).join(","));
  downloadBlob([columns.join(","),...rows].join("\r\n"),"text/csv","geo_labels.csv");
};
$("geojson-file").onchange=()=>busy(async()=>{const file=$("geojson-file").files[0];if(!file)return;if(file.size>20*1024**2)throw new Error("Use a GeoJSON file smaller than 20 MB.");const data=JSON.parse(await file.text());if(data.type!=="FeatureCollection"||!Array.isArray(data.features))throw new Error("Expected a GeoJSON FeatureCollection.");if(data.coordinate_space || (data.crs&&!/4326|CRS84/.test(JSON.stringify(data.crs))))throw new Error("Use WGS84 geographic coordinates for a map layer.");render(data);activatePanel("search-panel");setStatus("GeoJSON layer imported.");},"Reading map layer…");

async function refreshCheckpoints(){const data=await request("/api/checkpoints");$("checkpoint").replaceChildren();data.checkpoints.forEach(path=>$("checkpoint").add(new Option(path,path)));if(!data.checkpoints.length)$("checkpoint").add(new Option("Train a model first",""));}
$("refresh-checkpoints").onclick=()=>busy(refreshCheckpoints,"Refreshing checkpoints…");
async function pollTraining(){if(!trainingJob)return;try{const data=await request(`/api/train/${trainingJob}`);$("training-log").hidden=false;$("training-log").textContent=data.log||"Starting training…";if(data.status==="running")setTimeout(pollTraining,2500);else{trainingJob=null;$("cancel-training").hidden=true;setStatus(`Training ${data.status}. Outputs: ${data.output}`,data.status==="failed"?"error":"");await refreshCheckpoints();}}catch(error){setStatus(error.message,"error");}}
$("train-form").onsubmit=event=>{event.preventDefault();busy(async()=>{const result=await post("/api/train",{manifest:$("manifest").value,epochs:Number($("epochs").value),batch_size:Number($("batch-size").value)});trainingJob=result.job_id;$("cancel-training").hidden=false;setStatus("Training started.");pollTraining();},"Validating training data…",event.submitter);};
$("cancel-training").onclick=()=>busy(async()=>{const result=await request(`/api/train/${trainingJob}/cancel`,{method:"POST"});setStatus(result.message);},"Cancelling training…");
$("classify-form").onsubmit=event=>{event.preventDefault();busy(async()=>{const result=await request("/api/classify",{method:"POST",body:new FormData(event.target)});$("classification-result").hidden=false;$("classification-result").textContent=JSON.stringify(result,null,2);setStatus(`Predicted tile class: ${result.predicted_class}`);},"Classifying tile…",event.submitter);};

async function initialize(){
  const data=await request("/api/settings");applySettings(data,true);
  if(location.pathname==="/settings")activatePanel("settings-panel",false);
  await Promise.allSettled([refreshModels(),refreshCheckpoints(),request("/api/status").then(status=>$("gpu").textContent=status.gpu||"CPU mode")]);
  const running=sessionStorage.getItem("geo-active-job");if(running)busy(async()=>renderImages(await followJob(running)),"Reconnecting to analysis…");
}
initialize().catch(error=>setStatus(error.message,"error"));
