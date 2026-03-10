const DEFAULT_API_BASE = "https://fjrmhri-ta-berita-hoax-bertopic.hf.space";

function normalizeApiBaseUrl(raw) {
  const value = String(raw || "").trim();
  if (!value) {
    return "";
  }

  // Normalisasi domain HF Spaces: lowercase + underscore -> hyphen.
  const normalizedHfSpace = value.replace(
    /^https?:\/\/([^/]+)\.hf\.space/i,
    (_all, subdomain) =>
      `https://${String(subdomain).toLowerCase().replace(/_/g, "-")}.hf.space`,
  );

  try {
    const parsed = new URL(normalizedHfSpace);
    return parsed.origin.replace(/\/+$/, "");
  } catch (_error) {
    return normalizedHfSpace.replace(/\/+$/, "");
  }
}

function resolveApiBaseUrl() {
  const params = new URLSearchParams(window.location.search);
  const override = params.get("api");
  if (override && override.trim()) {
    return normalizeApiBaseUrl(override);
  }
  return normalizeApiBaseUrl(DEFAULT_API_BASE);
}

const API_BASE = resolveApiBaseUrl();
const ANALYZE_ENDPOINT = API_BASE ? `${API_BASE}/analyze` : "/analyze";
const HEALTH_ENDPOINT = API_BASE ? `${API_BASE}/health` : "/health";

const analyzeForm = document.getElementById("analyzeForm");
const inputText = document.getElementById("inputText");
const resetBtn = document.getElementById("resetBtn");
const copyBtn = document.getElementById("copyBtn");

const loader = document.getElementById("loader");
const errorBox = document.getElementById("errorBox");
const summaryPanel = document.getElementById("summaryPanel");
const highlightPanel = document.getElementById("highlightPanel");
const confidencePanel = document.getElementById("confidencePanel");
const topicPanel = document.getElementById("topicPanel");
const debugMeta = document.getElementById("debugMeta");

const highlightContent = document.getElementById("highlightContent");
const confidenceContent = document.getElementById("confidenceContent");
const topicContent = document.getElementById("topicContent");

const sumParagraphs = document.getElementById("sumParagraphs");
const sumSentences = document.getElementById("sumSentences");
const sumHoax = document.getElementById("sumHoax");
const sumFakta = document.getElementById("sumFakta");
const sumLowConf = document.getElementById("sumLowConf");

let latestResponse = null;
let latestHealth = null;

function escapeHtml(value) {
  return String(value)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#039;");
}

function pct(value) {
  return `${(Number(value) * 100).toFixed(2)}%`;
}

function getErrorMessage(detail, fallback) {
  if (typeof detail === "string" && detail.trim()) {
    const text = detail.trim();
    const lowered = text.toLowerCase();
    if (lowered.includes("<!doctype html") || lowered.includes("<html")) {
      return `Backend URL tidak valid atau endpoint tidak ditemukan. Pastikan API mengarah ke ${ANALYZE_ENDPOINT}`;
    }
    if (text.length > 300) {
      return `${text.slice(0, 300)}...`;
    }
    return text;
  }
  if (detail && typeof detail === "object") {
    if (typeof detail.message === "string" && detail.message.trim()) {
      return detail.message.trim();
    }
    return JSON.stringify(detail);
  }
  return fallback;
}

function setLoading(isLoading) {
  loader.classList.toggle("hidden", !isLoading);
}

function showError(message) {
  errorBox.textContent = message;
  errorBox.classList.remove("hidden");
}

function clearError() {
  errorBox.textContent = "";
  errorBox.classList.add("hidden");
}

function clearOutput() {
  latestResponse = null;
  latestHealth = null;
  summaryPanel.classList.add("hidden");
  highlightPanel.classList.add("hidden");
  confidencePanel.classList.add("hidden");
  topicPanel.classList.add("hidden");
  highlightContent.innerHTML = "";
  confidenceContent.innerHTML = "";
  topicContent.innerHTML = "";
  if (debugMeta) {
    debugMeta.textContent = "";
    debugMeta.classList.add("hidden");
  }
}

function renderSummary(summary) {
  sumParagraphs.textContent = String(summary.num_paragraphs ?? 0);
  sumSentences.textContent = String(summary.num_sentences ?? 0);
  sumHoax.textContent = String(summary.hoax_sentences ?? 0);
  sumFakta.textContent = String(summary.fakta_sentences ?? 0);
  sumLowConf.textContent = String(summary.low_conf_sentences ?? 0);
  summaryPanel.classList.remove("hidden");
}

function renderHighlights(paragraphs) {
  const blocks = paragraphs
    .map((paragraph) => {
      const spans = paragraph.sentences
        .map(
          (sentence) => `
            <span class="hl ${escapeHtml(sentence.color)}" title="${escapeHtml(
              `${sentence.label} | conf ${pct(sentence.confidence)} | Hoaks ${pct(sentence.prob_hoax)} | Fakta ${pct(
                sentence.prob_fakta,
              )}`,
            )}">${escapeHtml(sentence.text)}</span>
          `,
        )
        .join(" ");

      return `
        <article class="paragraph-block">
          <p class="paragraph-title">Paragraf ${paragraph.paragraph_index + 1}</p>
          <p class="paragraph-text">${spans || "<em>(Tidak ada kalimat terdeteksi)</em>"}</p>
        </article>
      `;
    })
    .join("");

  highlightContent.innerHTML = blocks;
  highlightPanel.classList.remove("hidden");
}

function renderConfidence(paragraphs) {
  const html = paragraphs
    .map((paragraph) => {
      const items = paragraph.sentences
        .map(
          (sentence) => `
            <div class="confidence-item">
              <div class="conf-left">
                <strong>[${escapeHtml(sentence.label)}]</strong> ${escapeHtml(sentence.text)}
              </div>
              <div class="conf-right">${pct(sentence.confidence)}</div>
            </div>
          `,
        )
        .join("");

      const summary = paragraph.paragraph_summary || {};
      return `
        <section class="confidence-block">
          <h3>Paragraf ${paragraph.paragraph_index + 1}</h3>
          ${items || "<p><em>(Tidak ada kalimat terdeteksi)</em></p>"}
          <p><small>Hoaks: ${summary.hoax_sentences ?? 0} | Fakta: ${summary.fakta_sentences ?? 0} | Avg conf: ${pct(
            summary.avg_confidence ?? 0,
          )} | Max hoaks prob: ${pct(summary.max_hoax_prob ?? 0)}</small></p>
        </section>
      `;
    })
    .join("");

  confidenceContent.innerHTML = html;
  confidencePanel.classList.remove("hidden");
}

function renderTopics(topics) {
  const items = Array.isArray(topics?.items) ? topics.items : [];
  if (!topics?.enabled || items.length === 0) {
    topicContent.innerHTML = "";
    topicPanel.classList.add("hidden");
    return;
  }

  topicContent.innerHTML = items
    .map(
      (topic) => `
        <article class="topic-card">
          <div class="topic-label">${escapeHtml(topic.topic_label ?? "Topik")}</div>
          <div class="topic-meta">ID: ${escapeHtml(topic.topic_id ?? "-")}</div>
          <div class="topic-meta">Skor: ${pct(topic.probability ?? 0)}</div>
        </article>
      `,
    )
    .join("");
  topicPanel.classList.remove("hidden");
}

async function fetchHealthData() {
  try {
    const response = await fetch(HEALTH_ENDPOINT, { method: "GET" });
    if (!response.ok) {
      return null;
    }
    const payload = await response.json().catch(() => null);
    return payload && typeof payload === "object" ? payload : null;
  } catch (_error) {
    return null;
  }
}

function renderDebugMeta(model, health) {
  if (!debugMeta) {
    return;
  }

  const source = health?.model_source ?? model?.source ?? "-";
  const threshold = health?.hoax_threshold ?? model?.hoax_threshold;
  const calibrationLoaded =
    health?.calibration_loaded ?? model?.calibration_loaded;
  const thresholdText =
    typeof threshold === "number" ? `${(threshold * 100).toFixed(1)}%` : "-";

  debugMeta.textContent = `Model source: ${source} | Hoaks threshold: ${thresholdText} | Calibration loaded: ${Boolean(calibrationLoaded)}`;
  debugMeta.classList.remove("hidden");
}

function formatCopyText(data) {
  const lines = [];
  lines.push("Ringkasan Analisis");
  lines.push(
    `Paragraf=${data.summary.num_paragraphs}, Kalimat=${data.summary.num_sentences}, Hoaks=${data.summary.hoax_sentences}, Fakta=${data.summary.fakta_sentences}, LowConf=${data.summary.low_conf_sentences}`,
  );
  lines.push("");

  for (const paragraph of data.paragraphs) {
    lines.push(`Paragraf ${paragraph.paragraph_index + 1}`);
    for (const sentence of paragraph.sentences) {
      lines.push(
        `- [${sentence.label}] conf=${pct(sentence.confidence)} :: ${sentence.text}`,
      );
    }
    lines.push("");
  }

  if (
    data.topics?.enabled &&
    Array.isArray(data.topics.items) &&
    data.topics.items.length > 0
  ) {
    lines.push("Topik");
    for (const topic of data.topics.items) {
      lines.push(
        `- ${topic.topic_label} (id=${topic.topic_id}, skor=${pct(topic.probability ?? 0)})`,
      );
    }
  }

  return lines.join("\n");
}

analyzeForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  clearError();
  clearOutput();

  const text = inputText.value.trim();
  if (!text) {
    showError("Input teks wajib diisi.");
    return;
  }

  setLoading(true);
  try {
    const response = await fetch(ANALYZE_ENDPOINT, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text }),
    });

    const rawBody = await response.text();
    let payload = {};
    if (rawBody) {
      try {
        payload = JSON.parse(rawBody);
      } catch (_error) {
        payload = { detail: rawBody };
      }
    }

    if (!response.ok) {
      const detail = getErrorMessage(payload.detail, `HTTP ${response.status}`);
      throw new Error(detail);
    }

    if (!payload || typeof payload !== "object") {
      throw new Error("Response API tidak valid.");
    }

    latestResponse = payload;
    latestHealth = await fetchHealthData();
    renderSummary(payload.summary || {});
    renderHighlights(payload.paragraphs || []);
    renderConfidence(payload.paragraphs || []);
    renderTopics(payload.topics || { enabled: false, items: [] });
    renderDebugMeta(payload.model || {}, latestHealth);
  } catch (error) {
    if (error instanceof TypeError) {
      showError(
        `Gagal menghubungi API (${ANALYZE_ENDPOINT}). Pastikan backend aktif dan CORS sudah sesuai.`,
      );
    } else {
      showError(`Gagal memproses: ${error.message}`);
    }
  } finally {
    setLoading(false);
  }
});

resetBtn.addEventListener("click", () => {
  clearError();
  clearOutput();
  inputText.value = "";
});

copyBtn.addEventListener("click", async () => {
  if (!latestResponse) {
    showError("Belum ada hasil untuk disalin.");
    return;
  }

  clearError();
  try {
    await navigator.clipboard.writeText(formatCopyText(latestResponse));
    window.alert("Hasil berhasil disalin.");
  } catch (error) {
    showError(`Gagal menyalin hasil: ${error.message}`);
  }
});
