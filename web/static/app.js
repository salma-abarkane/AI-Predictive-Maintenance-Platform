const form = document.getElementById("upload-form");
const fileInput = document.getElementById("dataset-file");
const feedback = document.getElementById("feedback");
const submitButton = document.getElementById("submit-button");
const resultsBody = document.getElementById("results-body");
const downloadLink = document.getElementById("download-link");
const dropzone = document.querySelector(".upload-dropzone");
const selectedFile = document.getElementById("selected-file");
const ollamaModelInput = document.getElementById("ollama-model");
const ollamaModelsList = document.getElementById("ollama-models");
const summaryBox = document.getElementById("summary-box");
const chatToggle = document.getElementById("chat-toggle");
const chatPanel = document.getElementById("chatbot-panel");
const chatClose = document.getElementById("chat-close");
const chatForm = document.getElementById("chat-form");
const chatInput = document.getElementById("chat-input");
const chatMessages = document.getElementById("chat-messages");

const rowsProcessed = document.getElementById("rows-processed");
const enginesDetected = document.getElementById("engines-detected");
const predictedAnomalies = document.getElementById("predicted-anomalies");
const criticalRows = document.getElementById("critical-rows");

loadOllamaModels();

["dragenter", "dragover"].forEach((eventName) => {
  dropzone.addEventListener(eventName, (event) => {
    event.preventDefault();
    dropzone.classList.add("drag-over");
  });
});

["dragleave", "drop"].forEach((eventName) => {
  dropzone.addEventListener(eventName, (event) => {
    event.preventDefault();
    dropzone.classList.remove("drag-over");
  });
});

dropzone.addEventListener("drop", (event) => {
  const files = event.dataTransfer.files;
  if (!files.length) {
    return;
  }
  fileInput.files = files;
  updateSelectedFileLabel();
});

fileInput.addEventListener("change", updateSelectedFileLabel);

chatToggle.addEventListener("click", () => {
  chatPanel.classList.toggle("hidden");
});

chatClose.addEventListener("click", () => {
  chatPanel.classList.add("hidden");
});

form.addEventListener("submit", async (event) => {
  event.preventDefault();

  if (!fileInput.files.length) {
    feedback.textContent = "Please choose a dataset first.";
    return;
  }

  const formData = new FormData();
  formData.append("file", fileInput.files[0]);
  formData.append("ollama_model", ollamaModelInput.value.trim() || "llama3.2");

  submitButton.disabled = true;
  submitButton.textContent = "Running prediction...";
  feedback.textContent = "Uploading dataset and computing predictions...";
  summaryBox.textContent = "Generating anomaly summary. The first Ollama response can take some time while the model loads.";

  try {
    const response = await fetch("/api/predict", {
      method: "POST",
      body: formData,
    });

    const payload = await response.json();
    if (!response.ok) {
      throw new Error(payload.detail || "Prediction failed.");
    }

    renderSummary(payload.summary);
    renderTable(payload.preview);
    renderLlmSummary(payload.llm_summary);
    downloadLink.href = payload.download_url;
    downloadLink.classList.remove("hidden");
    feedback.textContent = "Prediction completed successfully.";
  } catch (error) {
    feedback.textContent = error.message;
  } finally {
    submitButton.disabled = false;
    submitButton.textContent = "Run anomaly prediction";
  }
});

chatForm.addEventListener("submit", async (event) => {
  event.preventDefault();

  const message = chatInput.value.trim();
  if (!message) {
    return;
  }

  appendChatMessage("user", message);
  chatInput.value = "";
  appendChatMessage("assistant", "Thinking... The first Ollama answer can take a little while.");

  try {
    const response = await fetch("/api/chat", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        message,
        model: ollamaModelInput.value.trim() || "llama3.2",
      }),
    });

    const payload = await response.json();
    if (!response.ok) {
      throw new Error(payload.detail || "Chat request failed.");
    }

    replaceLastAssistantMessage(payload.text || "No response.");
  } catch (error) {
    replaceLastAssistantMessage(error.message);
  }
});

function renderSummary(summary) {
  rowsProcessed.textContent = formatNumber(summary.rows);
  enginesDetected.textContent = formatNumber(summary.engines);
  predictedAnomalies.textContent = formatNumber(summary.predicted_anomalies);
  criticalRows.textContent = formatNumber(summary.critical_rows);
}

function renderLlmSummary(summaryPayload) {
  if (!summaryPayload) {
    summaryBox.textContent = "No summary returned.";
    return;
  }
  summaryBox.textContent = summaryPayload.text;
}

function renderTable(rows) {
  if (!rows.length) {
    resultsBody.innerHTML = `
      <tr>
        <td colspan="6" class="empty-state">No rows returned.</td>
      </tr>
    `;
    return;
  }

  resultsBody.innerHTML = rows
    .map((row) => {
      const severityClass = row.severity.toLowerCase();
      const anomalyText = row.anomaly_prediction === 1 ? "Yes" : "No";
      return `
        <tr>
          <td>${row.unit_number}</td>
          <td>${row.time_in_cycles}</td>
          <td>${anomalyText}</td>
          <td>${Number(row.anomaly_probability).toFixed(4)}</td>
          <td><span class="badge ${severityClass}">${row.severity}</span></td>
          <td>${row.maintenance_window}</td>
        </tr>
      `;
    })
    .join("");
}

function formatNumber(value) {
  return new Intl.NumberFormat().format(value);
}

function updateSelectedFileLabel() {
  const file = fileInput.files[0];
  selectedFile.textContent = file ? file.name : "No file selected";
}

async function loadOllamaModels() {
  try {
    const response = await fetch("/api/ollama/models");
    const payload = await response.json();
    if (!payload.available) {
      return;
    }
    ollamaModelsList.innerHTML = payload.models
      .map((model) => `<option value="${model}"></option>`)
      .join("");
    if (payload.default_model) {
      ollamaModelInput.value = payload.default_model;
    }
  } catch (error) {
    console.error("Failed to load Ollama models", error);
  }
}

function appendChatMessage(role, text) {
  const article = document.createElement("article");
  article.className = `chat-message ${role}`;
  article.textContent = text;
  chatMessages.appendChild(article);
  chatMessages.scrollTop = chatMessages.scrollHeight;
}

function replaceLastAssistantMessage(text) {
  const messages = [...chatMessages.querySelectorAll(".chat-message.assistant")];
  const lastAssistant = messages[messages.length - 1];
  if (lastAssistant) {
    lastAssistant.textContent = text;
  }
  chatMessages.scrollTop = chatMessages.scrollHeight;
}
