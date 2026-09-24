let currentDocId = null;
let kbMode = false;

const $ = (id) => document.getElementById(id);

// Show current mode (kb / local / mock) and adjust UI accordingly.
fetch("/health")
  .then((r) => r.json())
  .then((d) => {
    $("mode").textContent = `mode: ${d.mode} (${d.region})`;
    kbMode = d.mode === "kb";
    if (kbMode) {
      // In KB mode the reports already live in the Knowledge Base (from S3),
      // so the user can ask directly without selecting/uploading.
      $("askBtn").disabled = false;
      $("uploadStatus").textContent =
        "Knowledge Base mode: reports already indexed from S3. Ask directly.";
    }
  })
  .catch(() => { $("mode").textContent = "mode: unknown"; });

// Load bundled sample reports into the dropdown.
fetch("/reports")
  .then((r) => r.json())
  .then((d) => {
    (d.reports || []).forEach((name) => {
      const opt = document.createElement("option");
      opt.value = name;
      opt.textContent = name;
      $("sampleSelect").appendChild(opt);
    });
  })
  .catch(() => {});

// Use a bundled sample report.
$("selectBtn").addEventListener("click", async () => {
  const name = $("sampleSelect").value;
  if (!name) {
    $("uploadStatus").textContent = "Pick a report from the list first.";
    return;
  }
  $("selectBtn").disabled = true;
  $("uploadStatus").textContent = "Indexing sample report…";
  const form = new FormData();
  form.append("name", name);
  try {
    const res = await fetch("/select-sample", { method: "POST", body: form });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || "Failed");
    currentDocId = data.doc_id;
    $("uploadStatus").textContent =
      `Ready: "${data.filename}" indexed into ${data.chunks} chunks.`;
    $("askBtn").disabled = false;
  } catch (err) {
    $("uploadStatus").textContent = "Error: " + err.message;
  } finally {
    $("selectBtn").disabled = false;
  }
});

// Upload a custom report.
$("uploadBtn").addEventListener("click", async () => {
  const file = $("fileInput").files[0];
  if (!file) {
    $("uploadStatus").textContent = "Please choose a .txt or .pdf file first.";
    return;
  }
  $("uploadBtn").disabled = true;
  $("uploadStatus").textContent = "Uploading and indexing…";
  const form = new FormData();
  form.append("file", file);
  try {
    const res = await fetch("/upload", { method: "POST", body: form });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || "Upload failed");
    currentDocId = data.doc_id;
    $("uploadStatus").textContent =
      `Indexed "${data.filename}" into ${data.chunks} chunks. You can ask now.`;
    $("askBtn").disabled = false;
  } catch (err) {
    $("uploadStatus").textContent = "Error: " + err.message;
  } finally {
    $("uploadBtn").disabled = false;
  }
});

// Ask a question.
$("askBtn").addEventListener("click", async () => {
  const question = $("questionInput").value.trim();
  if (!question) {
    $("askStatus").textContent = "Type a question first.";
    return;
  }
  if (!kbMode && !currentDocId) {
    $("askStatus").textContent = "Select or upload a report first.";
    return;
  }
  $("askBtn").disabled = true;
  $("askStatus").textContent = "Thinking…";

  const form = new FormData();
  form.append("question", question);
  if (currentDocId) form.append("doc_id", currentDocId);

  try {
    const res = await fetch("/ask", { method: "POST", body: form });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || "Ask failed");

    $("answerCard").style.display = "block";
    $("answer").textContent = data.answer;

    $("piiNote").textContent =
      data.pii_redacted && data.pii_redacted.length
        ? `⚠ PII redacted before model call: ${data.pii_redacted.join(", ")}`
        : "";

    const ev = $("evidence");
    ev.innerHTML = "";
    (data.evidence || []).forEach((chunk, i) => {
      const div = document.createElement("div");
      div.className = "chunk";
      const scoreLabel =
        chunk.score != null ? ` · score ${chunk.score}` : "";
      const sourceLabel = chunk.source
        ? `<div class="source">source: ${chunk.source}</div>`
        : "";
      div.innerHTML =
        `<span class="score">#${i + 1}${scoreLabel}</span>` +
        sourceLabel +
        "<div>" + chunk.text.replace(/</g, "&lt;") + "</div>";
      ev.appendChild(div);
    });

    $("askStatus").textContent = "";
  } catch (err) {
    $("askStatus").textContent = "Error: " + err.message;
  } finally {
    $("askBtn").disabled = false;
  }
});
