const API_BASE_URL = "https://fjrmhri-space-deteksi-hoax-ta.hf.space";

function resolveApiBaseUrl() {
  const params = new URLSearchParams(window.location.search);
  const override = params.get("api");
  if (override && override.trim()) {
    return override.trim().replace(/\/+$/, "");
  }
  return API_BASE_URL.replace(/\/+$/, "");
}

const API_BASE = resolveApiBaseUrl();
const ANALYZE_ENDPOINT = `${API_BASE}/analyze`;

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

const highlightContent = document.getElementById("highlightContent");
const confidenceContent = document.getElementById("confidenceContent");
const topicContent = document.getElementById("topicContent");

const sumParagraphs = document.getElementById("sumParagraphs");
const sumSentences = document.getElementById("sumSentences");
const sumHoax = document.getElementById("sumHoax");
const sumFakta = document.getElementById("sumFakta");
const sumLowConf = document.getElementById("sumLowConf");

let latestResponse = null;

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
  summaryPanel.classList.add("hidden");
  highlightPanel.classList.add("hidden");
  confidencePanel.classList.add("hidden");
  topicPanel.classList.add("hidden");
  highlightContent.innerHTML = "";
  confidenceContent.innerHTML = "";
  topicContent.innerHTML = "";
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
                sentence.prob_fakta
              )}`
            )}">${escapeHtml(sentence.text)}</span>
          `
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
          `
        )
        .join("");

      const summary = paragraph.paragraph_summary || {};
      return `
        <section class="confidence-block">
          <h3>Paragraf ${paragraph.paragraph_index + 1}</h3>
          ${items || "<p><em>(Tidak ada kalimat terdeteksi)</em></p>"}
          <p><small>Hoaks: ${summary.hoax_sentences ?? 0} | Fakta: ${summary.fakta_sentences ?? 0} | Avg conf: ${pct(
            summary.avg_confidence ?? 0
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
      `
    )
    .join("");
  topicPanel.classList.remove("hidden");
}

function formatCopyText(data) {
  const lines = [];
  lines.push("Ringkasan Analisis");
  lines.push(
    `Paragraf=${data.summary.num_paragraphs}, Kalimat=${data.summary.num_sentences}, Hoaks=${data.summary.hoax_sentences}, Fakta=${data.summary.fakta_sentences}, LowConf=${data.summary.low_conf_sentences}`
  );
  lines.push("");

  for (const paragraph of data.paragraphs) {
    lines.push(`Paragraf ${paragraph.paragraph_index + 1}`);
    for (const sentence of paragraph.sentences) {
      lines.push(`- [${sentence.label}] conf=${pct(sentence.confidence)} :: ${sentence.text}`);
    }
    lines.push("");
  }

  if (data.topics?.enabled && Array.isArray(data.topics.items) && data.topics.items.length > 0) {
    lines.push("Topik");
    for (const topic of data.topics.items) {
      lines.push(`- ${topic.topic_label} (id=${topic.topic_id}, skor=${pct(topic.probability ?? 0)})`);
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

    const payload = await response.json().catch(() => ({}));
    if (!response.ok) {
      throw new Error(payload.detail || `HTTP ${response.status}`);
    }

    latestResponse = payload;
    renderSummary(payload.summary || {});
    renderHighlights(payload.paragraphs || []);
    renderConfidence(payload.paragraphs || []);
    renderTopics(payload.topics || { enabled: false, items: [] });
  } catch (error) {
    showError(`Gagal memproses: ${error.message}`);
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
