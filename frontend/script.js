// ── DOM references ────────────────────────────────────────────────────────
const chatWindow = document.getElementById("chatWindow");
const userInput  = document.getElementById("userInput");
const sendBtn    = document.getElementById("sendBtn");
const statusBadge = document.getElementById("statusBadge");
const statusLabel = statusBadge.querySelector(".status-label");

// ── Emoji map for emotion badges ──────────────────────────────────────────
const EMOTION_EMOJI = {
  sadness:  "💙",
  joy:      "😊",
  love:     "💗",
  anger:    "🔥",
  fear:     "😰",
  surprise: "😲",
};

// ── Utility: current time as HH:MM ───────────────────────────────────────
function now() {
  return new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
}

// ── Scroll chat to the bottom ─────────────────────────────────────────────
function scrollToBottom() {
  chatWindow.scrollTop = chatWindow.scrollHeight;
}

// ── Render the welcome card once on page load ─────────────────────────────
function renderWelcome() {
  const card = document.createElement("div");
  card.className = "welcome-card";
  card.innerHTML = `
    <div class="welcome-icon">🧠</div>
    <h2>Welcome to MindEase</h2>
    <p>I'm here to listen and support you. Share how you're feeling — in any language. Everything you say stays private.</p>
  `;
  chatWindow.appendChild(card);
}

// ── Append a user message bubble ──────────────────────────────────────────
function appendUserMessage(text) {
  const row = document.createElement("div");
  row.className = "msg-row user";
  row.innerHTML = `
    <div class="bubble">${escapeHtml(text)}</div>
    <span class="msg-time">${now()}</span>
  `;
  chatWindow.appendChild(row);
  scrollToBottom();
}

// ── Append a bot message bubble with optional badges ─────────────────────
function appendBotMessage(text, intent, emotion, langName, langCode) {
  const row = document.createElement("div");
  row.className = "msg-row bot";

  // Build badges HTML
  let metaHtml = "";

  // Only show language badge if it's not English (English is the default)
  if (langCode && langCode !== "en") {
    metaHtml += `<span class="badge lang">🌐 ${escapeHtml(langName)}</span>`;
  }

  // Only show emotion badge on mental-health responses
  if (emotion) {
    const emoji = EMOTION_EMOJI[emotion] || "🎭";
    metaHtml += `<span class="badge emotion-${emotion}">${emoji} ${capitalise(emotion)}</span>`;
  }

  row.innerHTML = `
    <div class="bubble">${formatText(text)}</div>
    ${metaHtml ? `<div class="msg-meta">${metaHtml}</div>` : ""}
    <span class="msg-time">${now()}</span>
  `;
  chatWindow.appendChild(row);
  scrollToBottom();
}

// ── Show/remove the animated typing indicator ─────────────────────────────
function showTyping() {
  const row = document.createElement("div");
  row.className = "msg-row bot typing-row";
  row.id = "typingIndicator";
  row.innerHTML = `
    <div class="typing-bubble">
      <span class="typing-dot"></span>
      <span class="typing-dot"></span>
      <span class="typing-dot"></span>
    </div>
  `;
  chatWindow.appendChild(row);
  scrollToBottom();
}

function removeTyping() {
  const el = document.getElementById("typingIndicator");
  if (el) el.remove();
}

// ── Set the header status badge ───────────────────────────────────────────
function setStatus(state, label) {
  statusBadge.className = `status-badge ${state}`;
  statusLabel.textContent = label;
}

// ── Auto-grow the textarea as the user types ──────────────────────────────
userInput.addEventListener("input", () => {
  userInput.style.height = "auto";
  userInput.style.height = userInput.scrollHeight + "px";
  sendBtn.disabled = userInput.value.trim() === "";
});

// ── Send on Enter (Shift+Enter = newline) ─────────────────────────────────
userInput.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    if (!sendBtn.disabled) sendMessage();
  }
});

sendBtn.addEventListener("click", sendMessage);

// ── Core send function ────────────────────────────────────────────────────
async function sendMessage() {
  const text = userInput.value.trim();
  if (!text) return;

  // Clear + reset textarea
  userInput.value = "";
  userInput.style.height = "auto";
  sendBtn.disabled = true;

  appendUserMessage(text);
  showTyping();

  try {
    const res = await fetch("/chat", {
      method:  "POST",
      headers: { "Content-Type": "application/json" },
      body:    JSON.stringify({ message: text }),
    });

    removeTyping();

    if (!res.ok) {
      // Server returned an error (e.g. 503 models loading, 500 pipeline error)
      const err = await res.json().catch(() => ({ detail: "Unknown server error." }));
      appendBotMessage(
        `⚠️ ${err.detail || "Something went wrong. Please try again."}`,
        null, null, null, null
      );
      return;
    }

    const data = await res.json();
    // data = { response, intent, emotion, lang_name, lang_code }

    appendBotMessage(data.response, data.intent, data.emotion, data.lang_name, data.lang_code);

  } catch (networkErr) {
    // Fetch itself failed (server unreachable)
    removeTyping();
    appendBotMessage("⚠️ Cannot reach the server. Make sure uvicorn is running.", null, null, null, null);
    setStatus("error", "Offline");
  }
}

// ── Check server health once on load ─────────────────────────────────────
async function checkHealth() {
  try {
    const res  = await fetch("/health");
    const data = await res.json();
    if (data.models_loaded) {
      setStatus("online", "Ready");
      sendBtn.disabled = false;   // enable sending
      userInput.placeholder = "Share how you're feeling… (any language)";
    } else {
      setStatus("", "Loading models…");
      // Poll again in 3 seconds until models are ready
      setTimeout(checkHealth, 3000);
    }
  } catch {
    setStatus("error", "Offline");
    setTimeout(checkHealth, 5000);
  }
}

// ── Helpers ───────────────────────────────────────────────────────────────
function escapeHtml(str) {
  return str
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

// Convert newlines to <br> so multi-paragraph responses render correctly
function formatText(str) {
  return escapeHtml(str).replace(/\n/g, "<br>");
}

function capitalise(str) {
  return str.charAt(0).toUpperCase() + str.slice(1);
}

// ── Bootstrap ─────────────────────────────────────────────────────────────
renderWelcome();
checkHealth();
