// Global State
let wsSim = null;
let currentMode = "RL";
let burstActive = false;

// Canvas Topology State
const canvas = document.getElementById("topoCanvas");
const ctx = canvas.getContext("2d");
let particles = [];
let currentNodes = [];
let currentLinks = [];

// AI Conversation History Tracking
let lastLoggedSpeechTitle = "";
let lastLoggedStep = -1;

// ================= 1. WEBSOCKET & DASHBOARD LOGIC =================
function initSimSocket() {
    const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
    const url = `${protocol}//${window.location.host}/ws/telemetry`;
    wsSim = new WebSocket(url);

    wsSim.onopen = () => {
        document.getElementById("ws-status").textContent = "Canlı Simülasyon";
        document.getElementById("global-pulse").style.background = "#10b981";
    };

    wsSim.onmessage = (event) => {
        try {
            const data = JSON.parse(event.data);
            updateSimDashboard(data);
        } catch (e) {
            console.error("Telemetry parsing error:", e);
        }
    };

    wsSim.onclose = () => {
        document.getElementById("ws-status").textContent = "Yeniden Bağlanıyor...";
        document.getElementById("global-pulse").style.background = "#f59e0b";
        setTimeout(initSimSocket, 2000);
    };
}

function updateSimDashboard(data) {
    // 1. KPI Cards
    document.getElementById("val-power").textContent = data.total_power_w;
    const powerPercent = Math.min(100, (data.total_power_w / (data.max_power_w || 1)) * 100);
    document.getElementById("bar-power").style.width = `${powerPercent}%`;

    document.getElementById("val-savings").textContent = `${data.energy_savings_percent}%`;
    const savedWatts = Math.max(0, data.max_power_w - data.total_power_w);
    document.getElementById("val-carbon-text").textContent = `Karbon: ${(savedWatts * 0.4).toFixed(1)} g CO₂/saat`;

    document.getElementById("val-delay").textContent = data.avg_delay_ms;
    const qosBadge = document.getElementById("qos-meter");
    if (data.avg_delay_ms > 4.5 || data.packet_loss_percent > 1.0) {
        qosBadge.textContent = "SLA Riskli / Darboğaz";
        qosBadge.style.color = "#ef4444";
        qosBadge.style.background = "rgba(239, 68, 68, 0.15)";
    } else {
        qosBadge.textContent = "SLA Uyumlu (<5ms)";
        qosBadge.style.color = "#06b6d4";
        qosBadge.style.background = "rgba(6, 182, 212, 0.15)";
    }

    const activeCores = data.active_cores || [];
    const coreNodes = (data.nodes || []).filter(n => n.type === "core");
    const totalCores = coreNodes.length || activeCores.length || 4;
    document.getElementById("val-cores").textContent = `${activeCores.length} / ${totalCores}`;

    // Core pills
    const badgesRow = document.querySelector(".core-badges-row");
    if (badgesRow && coreNodes.length > 0) {
        badgesRow.innerHTML = coreNodes.map(c => {
            const isActive = activeCores.includes(c.id);
            const label = c.id.replace("c_", "C");
            return `<span class="core-pill ${isActive ? 'active' : 'sleep'}" title="${c.id}">${label}</span>`;
        }).join("");
    }

    document.getElementById("badge-step").textContent = `Adım: ${data.step}`;

    // 2. AI Conversational Speech & Reason Explainer ("Model Benle Konuşuyor")
    if (data.ai_speech) {
        updateAiSpeech(data.ai_speech, data.step);
    }

    // 3. Topology Data
    currentNodes = data.nodes || [];
    currentLinks = data.links || [];
    maybeSpawnParticles();
}

function updateAiSpeech(speech, step) {
    const heroCard = document.getElementById("ai-hero-card");
    const heroTitle = document.getElementById("ai-hero-title");
    const heroMsg = document.getElementById("ai-hero-message");
    const heroTime = document.getElementById("ai-hero-time");
    const metaDemand = document.getElementById("ai-meta-demand");
    const metaSavings = document.getElementById("ai-meta-savings");
    const metaCores = document.getElementById("ai-meta-cores");

    if (!heroCard || !heroMsg) return;

    // Update Hero Card
    heroCard.className = `ai-speech-hero ${speech.type || 'eco'}`;
    heroTitle.textContent = speech.title;
    heroMsg.textContent = speech.message;
    heroTime.textContent = speech.time;

    metaDemand.textContent = `${speech.demand_mbps} Mbps`;
    metaSavings.textContent = `%${speech.savings_pct}`;
    metaCores.textContent = (speech.active_cores || []).map(c => c.replace("c_", "C")).join(", ") || "Uykuda";

    // Add to Live Log Stream if title changed or every 4 steps
    const shouldLog = (speech.title !== lastLoggedSpeechTitle) || (step - lastLoggedStep >= 4);
    if (shouldLog) {
        lastLoggedSpeechTitle = speech.title;
        lastLoggedStep = step;
        appendAiLogItem(speech);
    }
}

function appendAiLogItem(speech) {
    const stream = document.getElementById("ai-log-stream");
    if (!stream) return;

    const item = document.createElement("div");
    item.className = `ai-log-item ${speech.type || ''}`;
    item.innerHTML = `
        <div class="ai-log-item-top">
            <span class="ai-log-item-title">${speech.title}</span>
            <span class="ai-log-item-time">${speech.time}</span>
        </div>
        <p class="ai-log-item-text">${speech.message}</p>
    `;

    stream.insertBefore(item, stream.firstChild);

    // Limit stream length to 20
    while (stream.children.length > 20) {
        stream.removeChild(stream.lastChild);
    }
}

function clearAiLog() {
    const stream = document.getElementById("ai-log-stream");
    if (stream) stream.innerHTML = "";
}

// ================= 2. CANVAS TOPOLOGY & FLOW PARTICLES =================
function maybeSpawnParticles() {
    if (particles.length > 70) return;
    currentLinks.forEach(link => {
        if (link.is_active && link.load > 20 && Math.random() < 0.3) {
            const srcNode = currentNodes.find(n => n.id === link.source);
            const tgtNode = currentNodes.find(n => n.id === link.target);
            if (srcNode && tgtNode) {
                particles.push({
                    x0: srcNode.x, y0: srcNode.y, x1: tgtNode.x, y1: tgtNode.y,
                    progress: 0, speed: 0.02 + Math.random() * 0.025,
                    color: link.utilization > 0.75 ? '#ef4444' : (link.utilization > 0.4 ? '#10b981' : '#38bdf8')
                });
            }
        }
    });
}

function renderTopology() {
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    const nodeMap = {};
    currentNodes.forEach(n => { nodeMap[n.id] = n; });

    // 1. Draw Links
    currentLinks.forEach(link => {
        const u = nodeMap[link.source];
        const v = nodeMap[link.target];
        if (!u || !v) return;

        ctx.beginPath();
        ctx.moveTo(u.x, u.y);
        ctx.lineTo(v.x, v.y);

        if (!link.is_active) {
            // Sleeping/inactive link
            ctx.strokeStyle = "rgba(51, 65, 85, 0.25)";
            ctx.lineWidth = 1;
            ctx.setLineDash([4, 4]);
            ctx.stroke();
            ctx.setLineDash([]);
        } else {
            // Active link with load gradient
            const util = link.utilization || 0;
            if (util > 0.8) {
                ctx.strokeStyle = "rgba(239, 68, 68, 0.9)";
                ctx.lineWidth = 3.5;
            } else if (util > 0.4) {
                ctx.strokeStyle = "rgba(16, 185, 129, 0.75)";
                ctx.lineWidth = 2.5;
            } else {
                ctx.strokeStyle = "rgba(56, 189, 248, 0.5)";
                ctx.lineWidth = 1.8;
            }
            ctx.stroke();
        }
    });

    // 2. Draw Moving Data Particles
    for (let i = particles.length - 1; i >= 0; i--) {
        const p = particles[i];
        p.progress += p.speed;
        if (p.progress >= 1.0) { particles.splice(i, 1); continue; }
        const px = p.x0 + (p.x1 - p.x0) * p.progress;
        const py = p.y0 + (p.y1 - p.y0) * p.progress;
        ctx.beginPath();
        ctx.arc(px, py, 3.2, 0, Math.PI * 2);
        ctx.fillStyle = p.color;
        ctx.fill();
    }

    // 3. Draw Nodes
    currentNodes.forEach(n => {
        ctx.beginPath();
        if (n.type === "core") {
            // Core Switch
            ctx.arc(n.x, n.y, 16, 0, Math.PI * 2);
            ctx.fillStyle = n.is_active ? "#10b981" : "#1e293b";
            ctx.fill();
            ctx.lineWidth = n.is_active ? 2.5 : 1.5;
            ctx.strokeStyle = n.is_active ? "#a7f3d0" : "#475569";
            ctx.stroke();

            // Label
            ctx.fillStyle = n.is_active ? "#ffffff" : "#94a3b8";
            ctx.font = "bold 10px JetBrains Mono";
            ctx.textAlign = "center";
            ctx.textBaseline = "middle";
            ctx.fillText(n.id.replace("c_", "C"), n.x, n.y);

            // Sleeping icon badge
            if (!n.is_active) {
                ctx.font = "8px Inter";
                ctx.fillStyle = "#64748b";
                ctx.fillText("🌙", n.x + 13, n.y - 10);
            }
        } else if (n.type === "aggregation") {
            // Aggregation Switch
            ctx.arc(n.x, n.y, 9, 0, Math.PI * 2);
            ctx.fillStyle = "#0284c7";
            ctx.fill();
            ctx.lineWidth = 1.5;
            ctx.strokeStyle = "#38bdf8";
            ctx.stroke();
        } else if (n.type === "edge") {
            // Edge Switch
            ctx.arc(n.x, n.y, 8, 0, Math.PI * 2);
            ctx.fillStyle = "#0369a1";
            ctx.fill();
            ctx.lineWidth = 1;
            ctx.strokeStyle = "#7dd3fc";
            ctx.stroke();
        } else if (n.type === "host") {
            // Host / Server Node
            ctx.rect(n.x - 4.5, n.y - 4.5, 9, 9);
            ctx.fillStyle = "#64748b";
            ctx.fill();
        }
    });

    requestAnimationFrame(renderTopology);
}

// ================= 3. USER CONTROLS & API =================
function setMode(mode) {
    fetch(`/api/mode/${mode}`, { method: 'POST' })
        .then(res => res.json())
        .then(data => {
            currentMode = data.mode;
            document.querySelectorAll('.mode-btn').forEach(b => b.classList.remove('active'));
            if (mode === 'RL') document.getElementById('btn-mode-rl')?.classList.add('active');
            if (mode === 'OSPF') document.getElementById('btn-mode-ospf')?.classList.add('active');
        });
}

function toggleBurst() {
    burstActive = !burstActive;
    fetch(`/api/burst/${burstActive}`, { method: 'POST' })
        .then(() => {
            const btn = document.getElementById("btn-burst");
            if (btn) {
                btn.classList.toggle("active", burstActive);
                document.getElementById("burst-label").textContent = burstActive ? "⚡ Patlama Aktif!" : "Ani Patlama (Burst)";
            }
        });
}

function onTrafficSliderInput(val) {
    document.getElementById("traffic-slider-val").textContent = `${val}%`;
    const intensity = parseFloat(val) / 100.0;
    fetch(`/api/traffic-intensity/${intensity}`, { method: 'POST' });
    updatePresetHighlight(intensity);
}

function onTrafficSliderChange(val) {
    const intensity = parseFloat(val) / 100.0;
    fetch(`/api/traffic-intensity/${intensity}`, { method: 'POST' });
    updatePresetHighlight(intensity);
}

function setTrafficPreset(intensity, label) {
    const val = Math.round(intensity * 100);
    const slider = document.getElementById("traffic-slider");
    if (slider) slider.value = val;
    document.getElementById("traffic-slider-val").textContent = `${val}% (${label})`;
    fetch(`/api/traffic-intensity/${intensity}`, { method: 'POST' });
    updatePresetHighlight(intensity);
}

function updatePresetHighlight(intensity) {
    document.querySelectorAll('.preset-btn').forEach(b => b.classList.remove('active'));
    if (intensity < 0.6) document.getElementById('preset-1')?.classList.add('active');
    else if (intensity < 1.1) document.getElementById('preset-2')?.classList.add('active');
    else if (intensity < 1.8) document.getElementById('preset-3')?.classList.add('active');
    else document.getElementById('preset-4')?.classList.add('active');
}

function changeTopology(topoName) {
    fetch(`/api/topology/${topoName}`, { method: 'POST' })
        .then(res => res.json())
        .then(data => {
            particles = [];
            currentNodes = [];
            currentLinks = [];
        });
}

// ================= 4. INITIALIZATION =================
window.addEventListener("DOMContentLoaded", () => {
    initSimSocket();
    requestAnimationFrame(renderTopology);
});
