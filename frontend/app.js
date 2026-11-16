const input = document.querySelector("#video-input");
const dropZone = document.querySelector("#drop-zone");
const inspectButton = document.querySelector("#inspect-button");
const selectedVideo = document.querySelector("#selected-video");
const busyOverlay = document.querySelector("#busy-overlay");
const toast = document.querySelector("#toast");
let selectedFile = null;
let toastTimer;

function showToast(message) {
  toast.textContent = message;
  toast.classList.add("show");
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => toast.classList.remove("show"), 4300);
}

function setFile(file) {
  if (!file) return;
  const supportedVideo = /\.(mp4|mov|m4v|avi|mkv|webm)$/i.test(file.name);
  if (!file.type.startsWith("video/") && !supportedVideo) return showToast("Choose a walkthrough video file.");
  if (file.size > 100 * 1024 * 1024) return showToast("This starter accepts videos up to 100 MB.");
  selectedFile = file;
  document.querySelector("#selected-name").textContent = file.name;
  document.querySelector("#selected-size").textContent = `${(file.size / 1024 / 1024).toFixed(1)} MB · Ready to review`;
  selectedVideo.hidden = false;
  dropZone.hidden = true;
  inspectButton.disabled = false;
}

function formatMoney(value, currency) {
  return new Intl.NumberFormat(undefined, { style: "currency", currency, maximumFractionDigits: 0 }).format(value);
}

function renderReview(data) {
  const { review, frames, estimate } = data;
  document.querySelector("#results-panel").hidden = false;
  document.querySelector("#scope-panel").hidden = false;
  document.querySelector("#review-summary").textContent = review.summary;
  const risk = document.querySelector("#risk-badge");
  risk.textContent = review.risk_level;
  risk.dataset.risk = review.risk_level.toLowerCase();

  const findings = document.querySelector("#finding-list");
  if (!review.defects.length) {
    findings.innerHTML = '<div class="empty-findings">No clear visible repair items were identified in these sampled frames. A closer on-site review may still find issues outside the video views.</div>';
  } else {
    findings.innerHTML = review.defects.map((defect) => `<article class="finding"><span class="finding-dot"></span><div class="finding-content"><h3>${escapeHtml(defect.category)} <span class="finding-location">${escapeHtml(defect.location)}</span></h3><p>${escapeHtml(defect.evidence)} ${escapeHtml(defect.next_step)}</p></div><span class="finding-score">${Math.round(defect.confidence * 100)}% · ${escapeHtml(defect.severity)}</span></article>`).join("");
  }

  document.querySelector("#frame-count-label").textContent = `${frames.length} frames · ${data.duration_seconds}s clip`;
  document.querySelector("#frame-strip").innerHTML = frames.map((frame) => `<div class="frame-card"><img src="${frame.preview}" alt="Walkthrough frame at ${frame.timestamp_seconds} seconds" /><span>${frame.timestamp_seconds}s</span></div>`).join("");
  const limitations = review.visual_limitations.length ? review.visual_limitations : ["A walkthrough shows visible surfaces only; concealed conditions were not assessed."];
  document.querySelector("#limitations-box").innerHTML = `<strong>Keep in mind</strong> · ${limitations.map(escapeHtml).join(" ")}`;

  const scope = document.querySelector("#scope-list");
  if (!review.scope.length) {
    scope.innerHTML = '<div class="empty-findings">No repair sequence was drafted from the visible findings.</div>';
  } else {
    scope.innerHTML = review.scope.map((phase, index) => `<article class="scope-phase"><span class="phase-number">${String(index + 1).padStart(2, "0")}</span><div class="phase-content"><h3>${escapeHtml(phase.phase)}</h3><ul>${phase.tasks.map((task) => `<li>${escapeHtml(task)}</li>`).join("")}</ul>${phase.depends_on.length ? `<div class="phase-dependency">After: ${phase.depends_on.map(escapeHtml).join(", ")}</div>` : ""}</div></article>`).join("");
  }

  document.querySelector("#estimate-placeholder").hidden = true;
  document.querySelector("#estimate-content").hidden = false;
  document.querySelector("#labor-value").textContent = `${estimate.labor_hours.low}–${estimate.labor_hours.high} hrs`;
  document.querySelector("#cost-value").textContent = `${formatMoney(estimate.material_cost.low, estimate.currency)}–${formatMoney(estimate.material_cost.high, estimate.currency)}`;
  document.querySelector("#duration-value").textContent = `${estimate.duration_days.low}–${estimate.duration_days.high} days`;
  document.querySelector("#estimate-basis").textContent = estimate.basis;
}

function escapeHtml(value) {
  return String(value).replace(/[&<>"']/g, (char) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[char]);
}

document.querySelector("#browse-button").addEventListener("click", (event) => { event.stopPropagation(); input.click(); });
dropZone.addEventListener("click", () => input.click());
dropZone.addEventListener("keydown", (event) => { if (event.key === "Enter" || event.key === " ") { event.preventDefault(); input.click(); } });
input.addEventListener("change", () => setFile(input.files[0]));
document.querySelector("#remove-video").addEventListener("click", () => {
  selectedFile = null;
  input.value = "";
  selectedVideo.hidden = true;
  dropZone.hidden = false;
  inspectButton.disabled = true;
});
for (const eventName of ["dragenter", "dragover"]) dropZone.addEventListener(eventName, (event) => { event.preventDefault(); dropZone.classList.add("dragover"); });
for (const eventName of ["dragleave", "drop"]) dropZone.addEventListener(eventName, (event) => { event.preventDefault(); dropZone.classList.remove("dragover"); });
dropZone.addEventListener("drop", (event) => setFile(event.dataTransfer.files[0]));

inspectButton.addEventListener("click", async () => {
  if (!selectedFile) return;
  const area = Number(document.querySelector("#area-sqm").value);
  if (!Number.isFinite(area) || area < 6) return showToast("Enter a floor area of at least 6 m².");
  const formData = new FormData();
  formData.append("video", selectedFile);
  formData.append("property_type", document.querySelector("#property-type").value);
  formData.append("area_sqm", String(area));
  formData.append("location", document.querySelector("#location").value);
  busyOverlay.hidden = false;
  inspectButton.disabled = true;
  try {
    const response = await fetch("/api/inspect", { method: "POST", body: formData });
    const data = await response.json();
    if (!response.ok) throw new Error(data.detail || "The inspection could not be completed.");
    renderReview(data);
    document.querySelector("#results-panel").scrollIntoView({ behavior: "smooth", block: "start" });
  } catch (error) {
    showToast(error.message);
  } finally {
    busyOverlay.hidden = true;
    inspectButton.disabled = !selectedFile;
  }
});

document.querySelector("#help-button")?.addEventListener("click", () => showToast("This first version samples video frames and drafts a visual triage brief. It is not a certified inspection."));

fetch("/api/health").then((response) => response.json()).then((data) => {
  const pill = document.querySelector("#connection-pill");
  const label = document.querySelector("#connection-label");
  if (data.visual_review_configured) {
    pill.classList.add("ready");
    label.textContent = "Vision review ready";
  } else {
    label.textContent = "Add vision API key";
  }
}).catch(() => { document.querySelector("#connection-label").textContent = "API unavailable"; });
