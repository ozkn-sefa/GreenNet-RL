// =========================================================================
// Green DCN Optimizer - Modern High-Definition Network Visualizer & AI Hub
// Tailored for Fat-Tree k=6 Data Center Topology and Real-time Green Routing
// =========================================================================

// Global State
let wsSim = null;
let currentMode = "RL";
let burstActive = false;
let activeTopologyType = "fattree_k6";
let isLinkLevelMode = false;
let isAiDrawerOpen = false;

// Canvas & Rendering State
const canvas = document.getElementById("topoCanvas");
const ctx = canvas.getContext("2d");
const canvasContainer = document.getElementById("canvas-container");
const tooltip = document.getElementById("node-tooltip");

let particles = [];
let currentNodes = [];
let currentLinks = [];
let scaledNodePositions = {};
let hoveredNode = null;

// City Names and Short Codes for SNDlib Abilene & WAN Nodes (Backward Compatibility)
const ABILENE_CITY_INFO = {
    "ATLAM5": { name: "Atlanta (M5)", code: "ATL-1" },
    "ATLAng": { name: "Atlanta", code: "ATL" },
    "CHINng": { name: "Chicago", code: "CHI" },
    "DNVRng": { name: "Denver", code: "DEN" },
    "HSTNng": { name: "Houston", code: "HOU" },
    "IPLSng": { name: "Indianapolis", code: "IND" },
    "KSCYng": { name: "Kansas City", code: "KSC" },
    "LOSAng": { name: "Los Angeles", code: "LAX" },
    "NYCMng": { name: "New York", code: "NYC" },
    "SNVAng": { name: "Sunnyvale", code: "SNV" },
    "STTLng": { name: "Seattle", code: "SEA" },
    "WASHng": { name: "Washington DC", code: "WDC" }
};

// AI Speech Tracking
let lastLoggedSpeechTitle = "";
let lastLoggedStep = -1;

// ================= 1. RESPONSIVE CANVAS SIZING =================
let resizeDebounce = null;

function resizeCanvas() {
    if (!canvasContainer || !canvas) return;
    const rect = canvasContainer.getBoundingClientRect();
    if (rect.width <= 0 || rect.height <= 0) return;

    const dpr = window.devicePixelRatio || 1;
    canvas.width = Math.floor(rect.width * dpr);
    canvas.height = Math.floor(rect.height * dpr);
    canvas.style.width = `${rect.width}px`;
    canvas.style.height = `${rect.height}px`;

    // Reset transform to exact DPR (prevents cumulative scaling bug on window resize)
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

    // Dinamik HUD yüksekliğini al
    const hudEl = document.querySelector(".hud-overlay-grid");
    const hudHeight = hudEl ? hudEl.getBoundingClientRect().height : 85;

    recomputeScaledPositions(rect.width, rect.height, hudHeight);
}

window.addEventListener("resize", () => {
    clearTimeout(resizeDebounce);
    resizeDebounce = setTimeout(resizeCanvas, 80);
});

// ================= 2. WEBSOCKET & DASHBOARD LOGIC =================
function initSimSocket() {
    const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
    const url = `${protocol}//${window.location.host}/ws/telemetry`;
    wsSim = new WebSocket(url);

    wsSim.onopen = () => {
        const statusEl = document.getElementById("ws-status");
        if (statusEl) statusEl.textContent = "Canlı Simülasyon (Bağlı)";
        const pulse = document.getElementById("global-pulse");
        if (pulse) pulse.style.background = "#10b981";
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
        const statusEl = document.getElementById("ws-status");
        if (statusEl) statusEl.textContent = "Yeniden Bağlanıyor...";
        const pulse = document.getElementById("global-pulse");
        if (pulse) pulse.style.background = "#f59e0b";
        setTimeout(initSimSocket, 2000);
    };
}

function updateSimDashboard(data) {
    activeTopologyType = data.topology_type || activeTopologyType;
    isLinkLevelMode = Boolean(data.is_link_level);

    if (data.burst_mode !== undefined) {
        burstActive = Boolean(data.burst_mode);
        const btnBurst = document.getElementById("btn-burst");
        if (btnBurst) {
            if (burstActive) btnBurst.classList.add("active");
            else btnBurst.classList.remove("active");
        }
    }

    // 0. HUD Card: Gelen İnternet / Ağ Trafik Hacmi & Proaktif Tahmin
    const demandMbps = Math.round(data.total_demand_mbps || 0);
    const predMbps = Math.round(data.predicted_next_demand_mbps || demandMbps);
    const growthRate = data.traffic_growth_rate || 0;
    const trafficEl = document.getElementById("val-traffic");
    const trafficUnitEl = document.getElementById("val-traffic-unit");
    if (trafficEl) {
        if (demandMbps >= 1000) {
            trafficEl.textContent = (demandMbps / 1000).toFixed(2);
            if (trafficUnitEl) trafficUnitEl.textContent = "Gbps";
        } else {
            trafficEl.textContent = demandMbps;
            if (trafficUnitEl) trafficUnitEl.textContent = "Mbps";
        }
    }

    const flowCountEl = document.getElementById("val-flow-count");
    if (flowCountEl) {
        const flowCount = (data.active_demands || []).length;
        flowCountEl.textContent = `${flowCount}`;
    }

    const predTrafficEl = document.getElementById("val-pred-traffic");
    if (predTrafficEl) {
        const arrow = growthRate >= 0.08 ? "↗" : (growthRate <= -0.08 ? "↘" : "→");
        const predStr = predMbps >= 1000 ? `${(predMbps / 1000).toFixed(2)} Gbps` : `${predMbps} Mbps`;
        predTrafficEl.textContent = `${arrow} ${predStr}`;
        predTrafficEl.style.color = growthRate >= 0.08 ? "#fbbf24" : "#34d399";
    }

    const barTraffic = document.getElementById("bar-traffic");
    if (barTraffic) {
        const trafficPct = Math.min(100, Math.max(5, (demandMbps / 8000) * 100));
        barTraffic.style.width = `${trafficPct}%`;
    }

    // 1. HUD Card 1: Power Consumption (OSPF Real Baseline vs RL Power)
    const powerW = Math.round(data.total_power_w || 0);
    const ospfPowerW = Math.round(data.ospf_power_w || powerW || 6780);
    const maxPowerW = Math.round(data.max_power_w || 9540);
    const powerEl = document.getElementById("val-power");
    if (powerEl) powerEl.textContent = powerW;
    
    const ospfRefEl = document.getElementById("val-ospf-ref");
    if (ospfRefEl) ospfRefEl.textContent = `${ospfPowerW} W`;

    const barPower = document.getElementById("bar-power");
    if (barPower) {
        const powerPct = Math.min(100, Math.max(10, (powerW / (ospfPowerW || 1)) * 100));
        barPower.style.width = `${powerPct}%`;
    }

    // 2. HUD Card 2: True Switch-Sleep Energy Savings (0% when 0 switches sleep!)
    const netSavingsW = Math.round(data.net_sleep_savings_w || 0);
    const netSavingsPct = data.net_sleep_savings_percent !== undefined ? data.net_sleep_savings_percent : 0;
    const totalEeeSavingsPct = data.energy_savings_percent || 0;

    const savingsEl = document.getElementById("val-savings");
    if (savingsEl) savingsEl.textContent = `${netSavingsPct}%`;

    const savingsDetailEl = document.getElementById("val-savings-detail");
    if (savingsDetailEl) {
        if (netSavingsW > 0) {
            savingsDetailEl.textContent = `Net -${netSavingsW} W (Donanım EEE Dahil: %${totalEeeSavingsPct})`;
        } else {
            savingsDetailEl.textContent = `0 W Switch Tasarrufu (Tüm Omurgalar Açık)`;
        }
    }

    // 3. HUD Card 3: Carbon Emissions & Grid Carbon Intensity
    const cm = data.carbon_metrics || {};
    const hourlyCO2 = cm.saved_carbon_vs_ospf_g_hr !== undefined ? cm.saved_carbon_vs_ospf_g_hr : (netSavingsW * 0.42).toFixed(1);
    const dailyCO2Kg = cm.saved_carbon_daily_kg !== undefined ? cm.saved_carbon_daily_kg : ((netSavingsW * 0.42 * 24) / 1000).toFixed(2);
    const gridCI = cm.carbon_intensity_g_kwh || 420;
    
    const carbonHourlyEl = document.getElementById("val-carbon-hourly");
    if (carbonHourlyEl) carbonHourlyEl.innerHTML = `${hourlyCO2} <small>g CO₂/saat</small>`;
    const carbonDailyEl = document.getElementById("val-carbon-daily");
    if (carbonDailyEl) carbonDailyEl.textContent = `Günde ~${dailyCO2Kg} kg CO₂ önleniyor`;
    const carbonIntensityEl = document.getElementById("val-carbon-intensity");
    if (carbonIntensityEl) carbonIntensityEl.textContent = `Şebeke Karbon Yoğunluğu: ${gridCI} g/kWh`;

    // 4. HUD Card 4: Delay, QoS & Switch Wake-Up Synchronization Delay
    const delayEl = document.getElementById("val-delay");
    if (delayEl) delayEl.textContent = (data.avg_delay_ms || 0.0).toFixed(2);

    const lossEl = document.getElementById("val-loss");
    if (lossEl) lossEl.textContent = `${(data.packet_loss_percent || 0.0).toFixed(2)}%`;

    const wakeupDelayMs = data.wakeup_delay_ms || 0;
    const wakeupDelayEl = document.getElementById("val-wakeup-delay");
    if (wakeupDelayEl) {
        if (wakeupDelayMs > 0) {
            wakeupDelayEl.textContent = `| ⏳ Uyanma: +${wakeupDelayMs.toFixed(2)}ms`;
            wakeupDelayEl.style.color = "#fbbf24";
        } else {
            wakeupDelayEl.textContent = "";
        }
    }

    const qosBadge = document.getElementById("qos-meter");
    if (qosBadge) {
        if ((data.avg_delay_ms || 0) > 4.5 || (data.packet_loss_percent || 0) > 0.5) {
            qosBadge.textContent = "SLA Riskli / Darboğaz";
            qosBadge.className = "qos-pill warn";
        } else if (wakeupDelayMs > 0) {
            qosBadge.textContent = "⏳ Switch Uyanıyor (Sync)";
            qosBadge.className = "qos-pill ok";
        } else {
            qosBadge.textContent = "SLA Uyumlu (<3.5ms)";
            qosBadge.className = "qos-pill ok";
        }
    }

    // 5. HUD Card 5: Core Status & Pills
    const coreNodes = (data.nodes || []).filter(n => n.type === "core");
    const activeCores = data.active_cores || [];
    const wakingCores = data.waking_cores || [];
    const totalCores = coreNodes.length || activeCores.length || (activeTopologyType === "fattree_k6" ? 9 : 4);
    const activeCount = activeCores.length;
    const sleepingCount = totalCores - activeCount;

    const valCoresEl = document.getElementById("val-cores");
    if (valCoresEl) valCoresEl.textContent = `${activeCount} / ${totalCores} Aktif`;

    const sleepCountEl = document.getElementById("val-sleep-count");
    if (sleepCountEl) sleepCountEl.textContent = `${sleepingCount} Omurga Uykuda`;

    const badgesRow = document.querySelector(".core-badges-row");
    if (badgesRow && coreNodes.length > 0) {
        badgesRow.innerHTML = coreNodes.map(c => {
            const isActive = activeCores.includes(c.id);
            const isWaking = wakingCores.includes(c.id);
            const label = c.id.replace("c_", "C") + (isWaking ? "⏳" : "");
            return `<span class="core-pill ${isActive ? 'active' : 'sleep'}" title="${c.id} - ${isWaking ? 'Uyanıyor (20sn Sync)' : (isActive ? 'Aktif' : 'Uyku Modu')}">${label}</span>`;
        }).join("");
    }

    // 6. AI Karar Asistanı & Konuşma Hero Kartı
    if (data.ai_speech) {
        const heroBadge = document.getElementById("ai-hero-badge");
        const heroTitle = document.getElementById("ai-hero-title");
        const heroTime = document.getElementById("ai-hero-time");
        const heroMsg = document.getElementById("ai-hero-message");
        const metaDemand = document.getElementById("ai-meta-demand");
        const metaSavings = document.getElementById("ai-meta-savings");
        const metaCores = document.getElementById("ai-meta-cores");

        if (heroTitle) heroTitle.textContent = data.ai_speech.title || "Model Analiz Ediyor";
        if (heroTime) heroTime.textContent = data.ai_speech.time || "--:--:--";
        if (heroMsg) heroMsg.textContent = data.ai_speech.message || "";
        if (metaDemand) metaDemand.textContent = `${Math.round(data.ai_speech.demand_mbps || 0)} Mbps`;
        if (metaSavings) metaSavings.textContent = `%${data.ai_speech.savings_pct || 0}`;
        
        if (metaCores) {
            metaCores.textContent = `${activeCount} / ${totalCores} Aktif`;
        }

        const stepBadge = document.getElementById("badge-step");
        if (stepBadge && data.step !== undefined) stepBadge.textContent = `Adım: ${data.step}`;

        // Stream Log Ekleme
        if (data.ai_speech.title !== lastLoggedSpeechTitle || (data.step - lastLoggedStep) >= 5) {
            addAiDecisionStreamCard(data.ai_speech, data.step);
            lastLoggedSpeechTitle = data.ai_speech.title;
            lastLoggedStep = data.step;
        }
    }

    // 7. Update Topology Data for Drawing
    currentNodes = data.nodes || [];
    currentLinks = data.links || [];

    const rect = canvasContainer.getBoundingClientRect();
    recomputeScaledPositions(rect.width, rect.height);
}

function addAiDecisionStreamCard(speech, step) {
    const container = document.getElementById("ai-log-stream");
    if (!container) return;

    const card = document.createElement("div");
    card.className = `stream-card ${speech.type || 'eco'}`;

    card.innerHTML = `
        <div class="stream-card-top">
            <span class="stream-title">${speech.title || 'Karar'}</span>
            <span class="stream-time">${speech.time || ''}</span>
        </div>
        <p class="stream-msg">${speech.message || ''}</p>
    `;

    container.insertBefore(card, container.firstChild);

    // En fazla 20 kart tut
    while (container.children.length > 20) {
        container.removeChild(container.lastChild);
    }
}

function clearAiLog() {
    const container = document.getElementById("ai-log-stream");
    if (container) container.innerHTML = "";
}

function toggleAiDrawer() {
    const drawer = document.getElementById("ai-drawer");
    const btn = document.getElementById("btn-toggle-ai");
    if (!drawer) return;

    isAiDrawerOpen = !isAiDrawerOpen;
    if (isAiDrawerOpen) {
        drawer.classList.add("open");
        if (btn) btn.classList.add("active");
    } else {
        drawer.classList.remove("open");
        if (btn) btn.classList.remove("active");
    }
}

// ================= 3. TOPOLOGY COORDINATE SCALING =================
function recomputeScaledPositions(width, height, hudHeight) {
    if (!currentNodes.length) return;

    if (!hudHeight) {
        const hudEl = document.querySelector(".hud-overlay-grid");
        hudHeight = hudEl ? hudEl.getBoundingClientRect().height : 85;
    }

    const isFatTree = activeTopologyType.includes("fattree") || activeTopologyType.includes("k4") || activeTopologyType.includes("k6");

    if (isFatTree) {
        layoutFatTree(width, height, hudHeight);
    } else {
        layoutWAN(width, height, hudHeight);
    }
}

function layoutFatTree(width, height, hudHeight) {
    const paddingX = Math.max(30, Math.min(60, width * 0.04));
    // HUD kartlarının tam altından güvenli pay bırak
    const topMargin = Math.max(90, (hudHeight || 85) + 30);
    const bottomMargin = 45;
    const usableWidth = Math.max(300, width - (paddingX * 2));
    const usableHeight = Math.max(250, height - topMargin - bottomMargin);

    const coreNodes = currentNodes.filter(n => n.type === "core");
    const aggNodes = currentNodes.filter(n => n.type === "aggregation");
    const edgeNodes = currentNodes.filter(n => n.type === "edge");
    const hostNodes = currentNodes.filter(n => n.type === "host");

    // Responsive Düğüm Yarıçapları (Genişliğe ve yüksekliğe göre akıllı küçülür)
    const coreRadius = Math.max(12, Math.min(20, width / 70));
    const aggRadius = Math.max(8, Math.min(14, width / 95));
    const edgeRadius = Math.max(7, Math.min(13, width / 105));
    const hostRadius = Math.max(4, Math.min(7, width / 180));

    // Y Katmanları
    const yCore = topMargin + usableHeight * 0.08;
    const yAgg = topMargin + usableHeight * 0.38;
    const yEdge = topMargin + usableHeight * 0.68;
    const yHost = topMargin + usableHeight * 0.95;

    // 1. Core Switches (9 adet for k=6)
    const coreStep = usableWidth / (coreNodes.length + 1);
    coreNodes.forEach((node, idx) => {
        scaledNodePositions[node.id] = {
            x: paddingX + (idx + 1) * coreStep,
            y: yCore,
            radius: coreRadius,
            type: "core",
            label: node.id.replace("c_", "C")
        };
    });

    // 2. Pod Gruplaması (Aggregation & Edge)
    const k = activeTopologyType.includes("k6") ? 6 : 4;
    const podWidth = usableWidth / k;

    // Pod bazında Agg ve Edge
    aggNodes.forEach(node => {
        const parts = node.id.split("_");
        const pod = parseInt(parts[1]) || 0;
        const aggIdx = parseInt(parts[2]) || 0;
        const countInPod = k / 2;
        const stepInPod = podWidth / (countInPod + 1);
        
        scaledNodePositions[node.id] = {
            x: paddingX + pod * podWidth + (aggIdx + 1) * stepInPod,
            y: yAgg,
            radius: aggRadius,
            type: "aggregation",
            label: `A${pod}_${aggIdx}`
        };
    });

    edgeNodes.forEach(node => {
        const parts = node.id.split("_");
        const pod = parseInt(parts[1]) || 0;
        const edgeIdx = parseInt(parts[2]) || 0;
        const countInPod = k / 2;
        const stepInPod = podWidth / (countInPod + 1);

        scaledNodePositions[node.id] = {
            x: paddingX + pod * podWidth + (edgeIdx + 1) * stepInPod,
            y: yEdge,
            radius: edgeRadius,
            type: "edge",
            label: `E${pod}_${edgeIdx}`
        };
    });

    // 3. Hosts (54 sunucu for k=6)
    const hostStep = usableWidth / (hostNodes.length + 1);
    hostNodes.forEach((node, idx) => {
        scaledNodePositions[node.id] = {
            x: paddingX + (idx + 1) * hostStep,
            y: yHost,
            radius: hostRadius,
            type: "host",
            label: `H${idx + 1}`
        };
    });
}

function layoutWAN(width, height, hudHeight) {
    let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
    currentNodes.forEach(n => {
        if (n.x < minX) minX = n.x;
        if (n.x > maxX) maxX = n.x;
        if (n.y < minY) minY = n.y;
        if (n.y > maxY) maxY = n.y;
    });

    const rangeX = maxX - minX || 1;
    const rangeY = maxY - minY || 1;
    const paddingX = Math.max(30, width * 0.05);
    const topMargin = Math.max(90, (hudHeight || 85) + 30);
    const bottomMargin = 50;
    const usableW = width - paddingX * 2;
    const usableH = height - topMargin - bottomMargin;

    currentNodes.forEach(n => {
        const normX = (n.x - minX) / rangeX;
        const normY = (n.y - minY) / rangeY;
        scaledNodePositions[n.id] = {
            x: paddingX + normX * usableW,
            y: topMargin + normY * usableH,
            radius: n.type === "host" ? 5 : Math.max(12, Math.min(18, width / 75)),
            type: n.type,
            label: n.label || n.id
        };
    });
}

// ================= 4. HIGH-DEF SPHERE (3D KÜRE) RENDERER =================
function drawSphereNode(x, y, radius, colorTheme, label, isActive, isCore, nodeType, isWaking) {
    ctx.save();

    // 1. Düğümün Altındaki Hafif Gölge (Drop Shadow)
    ctx.beginPath();
    ctx.ellipse(x, y + radius * 0.9, radius * 0.9, radius * 0.35, 0, 0, Math.PI * 2);
    ctx.fillStyle = "rgba(0, 0, 0, 0.45)";
    ctx.fill();

    // 2. 3D Metalik Küre Gradyanı (Kullanıcının Görselindeki Küreler Gibi)
    const grad = ctx.createRadialGradient(
        x - radius * 0.35, y - radius * 0.35, radius * 0.1,
        x, y, radius
    );

    if (isCore) {
        if (isWaking) {
            // Uyanmakta Olan (Boot / OpenFlow Sync) Kehribar-Turkuaz Küre
            grad.addColorStop(0, "#fef3c7");
            grad.addColorStop(0.35, "#fbbf24");
            grad.addColorStop(0.8, "#d97706");
            grad.addColorStop(1, "#78350f");
        } else if (isActive) {
            // Aktif Yeşil Core Küresi
            grad.addColorStop(0, "#d1fae5");
            grad.addColorStop(0.3, "#34d399");
            grad.addColorStop(0.8, "#059669");
            grad.addColorStop(1, "#064e3b");
        } else {
            // Uykudaki Eko-Koyu Core Küresi (Sleep Mode)
            grad.addColorStop(0, "#64748b");
            grad.addColorStop(0.4, "#334155");
            grad.addColorStop(0.85, "#1e293b");
            grad.addColorStop(1, "#0f172a");
        }
    } else if (nodeType === "aggregation") {
        // Aggregation Küresi (Mavi/Çelik Tonu)
        grad.addColorStop(0, "#e0f2fe");
        grad.addColorStop(0.3, "#38bdf8");
        grad.addColorStop(0.8, "#0284c7");
        grad.addColorStop(1, "#075985");
    } else if (nodeType === "edge") {
        // Edge Küresi (Gümüş / Metalik Şık Küre)
        grad.addColorStop(0, "#f8fafc");
        grad.addColorStop(0.35, "#94a3b8");
        grad.addColorStop(0.8, "#475569");
        grad.addColorStop(1, "#1e293b");
    } else {
        // Sunucu / Host Küresi
        grad.addColorStop(0, "#cbd5e1");
        grad.addColorStop(0.5, "#64748b");
        grad.addColorStop(1, "#334155");
    }

    // Küre Gövdesi
    ctx.beginPath();
    ctx.arc(x, y, radius, 0, Math.PI * 2);
    ctx.fillStyle = grad;
    ctx.fill();

    // Dış İnce Parlama Halkası
    ctx.lineWidth = isActive ? 1.8 : 1.0;
    ctx.strokeStyle = isWaking ? "#fde68a" : (isActive ? (isCore ? "rgba(110, 231, 183, 0.8)" : "rgba(255, 255, 255, 0.4)") : "rgba(255, 255, 255, 0.15)");
    ctx.stroke();

    // 3. Kullanıcının Resmindeki Küçük Bağlantı Portu Noktaları (Port Squares)
    if (nodeType !== "host") {
        const portOffset = radius * 0.95;
        const portSize = 3.5;
        ctx.fillStyle = "#0f172a";

        // Üst, Alt, Sol, Sağ port noktaları
        [[x, y - portOffset], [x, y + portOffset], [x - portOffset, y], [x + portOffset, y]].forEach(([px, py]) => {
            ctx.fillRect(px - portSize / 2, py - portSize / 2, portSize, portSize);
        });
    }

    // 4. Kullanıcının Resmindeki 'P' veya 'C' Kutucukları (Badge Box)
    if (isCore) {
        // Core Switch'in Üstünde Şık 'C' Kutucuğu
        const badgeY = y - radius - 11;
        ctx.beginPath();
        ctx.rect(x - 8, badgeY, 16, 12);
        ctx.fillStyle = isWaking ? "rgba(245, 158, 11, 0.35)" : (isActive ? "rgba(16, 185, 129, 0.25)" : "rgba(51, 65, 85, 0.5)");
        ctx.fill();
        ctx.strokeStyle = isWaking ? "#fbbf24" : (isActive ? "#10b981" : "#64748b");
        ctx.lineWidth = 1;
        ctx.stroke();

        ctx.font = "bold 8px JetBrains Mono";
        ctx.fillStyle = isWaking ? "#fef3c7" : (isActive ? "#a7f3d0" : "#94a3b8");
        ctx.textAlign = "center";
        ctx.textBaseline = "middle";
        ctx.fillText("C", x, badgeY + 6);
    } else if (nodeType === "aggregation" || nodeType === "edge") {
        // Pod Switch'lerin Altında 'P' Kutucuğu
        const badgeY = y + radius + 4;
        ctx.beginPath();
        ctx.rect(x - 7, badgeY, 14, 11);
        ctx.fillStyle = "rgba(15, 23, 42, 0.6)";
        ctx.fill();
        ctx.strokeStyle = "rgba(255, 255, 255, 0.2)";
        ctx.lineWidth = 0.8;
        ctx.stroke();

        ctx.font = "7px JetBrains Mono";
        ctx.fillStyle = "#94a3b8";
        ctx.textAlign = "center";
        ctx.textBaseline = "middle";
        ctx.fillText("P", x, badgeY + 5.5);
    }

    // 5. Küre İçi Etiket (C0, C1 .. veya Resimdeki Numaralar)
    if (radius > 9) {
        ctx.font = isCore ? "bold 10px JetBrains Mono" : "bold 9px JetBrains Mono";
        ctx.fillStyle = isActive ? (isCore ? "#064e3b" : "#0f172a") : "#cbd5e1";
        ctx.textAlign = "center";
        ctx.textBaseline = "middle";
        ctx.fillText(label || "", x, y);
    }

    // 6. Uykudaki veya Uyanmakta Olan Core İçin Rozet (🌙 veya ⏳)
    if (isCore && (!isActive || isWaking)) {
        ctx.beginPath();
        ctx.arc(x + radius * 0.75, y - radius * 0.75, 9, 0, Math.PI * 2);
        ctx.fillStyle = "rgba(15, 23, 42, 0.95)";
        ctx.fill();
        ctx.strokeStyle = isWaking ? "#38bdf8" : "#f59e0b";
        ctx.lineWidth = 1.2;
        ctx.stroke();

        ctx.font = "9px Inter";
        ctx.fillStyle = isWaking ? "#38bdf8" : "#fbbf24";
        ctx.textAlign = "center";
        ctx.textBaseline = "middle";
        ctx.fillText(isWaking ? "⏳" : "🌙", x + radius * 0.75, y - radius * 0.75);
    }

    ctx.restore();
}

// ================= 5. PARTICLE & TOPOLOGY RENDER LOOP =================
function spawnParticles() {
    if (particles.length > 70) return;
    currentLinks.forEach(link => {
        if (link.is_active && !link.is_sleeping && link.load > 8 && Math.random() < 0.25) {
            const u = scaledNodePositions[link.source];
            const v = scaledNodePositions[link.target];
            if (u && v) {
                particles.push({
                    x0: u.x, y0: u.y, x1: v.x, y1: v.y,
                    progress: 0,
                    speed: 0.012 + Math.random() * 0.02,
                    color: link.utilization > 0.8 ? '#ef4444' : (link.utilization > 0.4 ? '#f59e0b' : '#10b981')
                });
            }
        }
    });
}

function renderTopology() {
    const rect = canvasContainer ? canvasContainer.getBoundingClientRect() : { width: canvas.width, height: canvas.height };
    ctx.clearRect(0, 0, rect.width, rect.height);

    // 1. Hatları (Links) Çiz
    currentLinks.forEach(link => {
        const u = scaledNodePositions[link.source];
        const v = scaledNodePositions[link.target];
        if (!u || !v) return;

        const isSleeping = link.is_sleeping;
        const isActive = link.is_active && !isSleeping;

        ctx.beginPath();
        ctx.moveTo(u.x, u.y);
        ctx.lineTo(v.x, v.y);

        if (isSleeping) {
            // Uyuyan Hat (EEE / Low Power)
            ctx.strokeStyle = "rgba(245, 158, 11, 0.4)";
            ctx.lineWidth = 1.6;
            ctx.setLineDash([5, 5]);
            ctx.stroke();
            ctx.setLineDash([]);
        } else if (!isActive) {
            // Devre Dışı Hat
            ctx.strokeStyle = "rgba(51, 65, 85, 0.2)";
            ctx.lineWidth = 1.0;
            ctx.setLineDash([4, 4]);
            ctx.stroke();
            ctx.setLineDash([]);
        } else {
            // Aktif Hat - Trafik Yoğunluğuna Göre Renk & Kalınlık
            const util = link.utilization || 0;
            if (util > 0.8) {
                ctx.strokeStyle = "rgba(239, 68, 68, 0.85)";
                ctx.lineWidth = 3.2;
            } else if (util > 0.4) {
                ctx.strokeStyle = "rgba(245, 158, 11, 0.8)";
                ctx.lineWidth = 2.4;
            } else {
                ctx.strokeStyle = "rgba(16, 185, 129, 0.65)";
                ctx.lineWidth = 1.6;
            }
            ctx.stroke();
        }
    });

    // 2. Akan Veri Parçacıklarını Çiz
    spawnParticles();
    for (let i = particles.length - 1; i >= 0; i--) {
        const p = particles[i];
        p.progress += p.speed;
        if (p.progress >= 1.0) {
            particles.splice(i, 1);
            continue;
        }
        const px = p.x0 + (p.x1 - p.x0) * p.progress;
        const py = p.y0 + (p.y1 - p.y0) * p.progress;

        ctx.beginPath();
        ctx.arc(px, py, 3.0, 0, Math.PI * 2);
        ctx.fillStyle = p.color;
        ctx.shadowColor = p.color;
        ctx.shadowBlur = 6;
        ctx.fill();
        ctx.shadowBlur = 0;
    }

    // 3. Düğümleri (Küreleri) Çiz
    currentNodes.forEach(node => {
        const pos = scaledNodePositions[node.id];
        if (!pos) return;

        const isCore = node.type === "core";
        const isActive = node.is_active;
        const isWaking = Boolean(node.is_waking);
        drawSphereNode(pos.x, pos.y, pos.radius, null, pos.label, isActive, isCore, node.type, isWaking);
    });

    requestAnimationFrame(renderTopology);
}

// ================= 6. TOOLTIP & MOUSE INTERACTION =================
if (canvas) {
    canvas.addEventListener("mousemove", (e) => {
        const rect = canvas.getBoundingClientRect();
        const mouseX = e.clientX - rect.left;
        const mouseY = e.clientY - rect.top;

        let found = null;
        for (const [id, pos] of Object.entries(scaledNodePositions)) {
            const dist = Math.hypot(mouseX - pos.x, mouseY - pos.y);
            if (dist <= pos.radius + 4) {
                found = currentNodes.find(n => n.id === id);
                break;
            }
        }

        hoveredNode = found;
        if (found && tooltip) {
            const pos = scaledNodePositions[found.id];
            tooltip.style.display = "block";
            tooltip.style.left = `${pos.x + 15}px`;
            tooltip.style.top = `${pos.y - 15}px`;

            const typeLabel = found.type === "core" ? "Katman 3: Omurga (Core)" :
                              (found.type === "aggregation" ? "Katman 2: Dağıtım (Agg)" :
                              (found.type === "edge" ? "Katman 1: Kenar (ToR)" : "Katman 0: Sunucu (Host)"));

            const stateStr = found.is_waking ? "⏳ Uyanıyor (20sn Boot & Sync)" :
                             (found.is_active ? "● Aktif (Çalışıyor)" : "🌙 Uyku Modu (Sleep)");
            const stateCol = found.is_waking ? "#fbbf24" : (found.is_active ? "#34d399" : "#f59e0b");

            tooltip.innerHTML = `
                <div style="font-weight:700; color:#38bdf8; margin-bottom:2px;">${found.label || found.id}</div>
                <div style="font-size:0.7rem; color:#94a3b8; margin-bottom:4px;">${typeLabel}</div>
                <div>Durum: <strong style="color:${stateCol}">${stateStr}</strong></div>
            `;
        } else if (tooltip) {
            tooltip.style.display = "none";
        }
    });

    canvas.addEventListener("mouseleave", () => {
        if (tooltip) tooltip.style.display = "none";
        hoveredNode = null;
    });
}

let savedSliderBeforeBurst = 85;

// ================= 7. KONTROLLER & API ÇAĞRILARI =================
function setMode(mode) {
    fetch(`/api/mode/${mode}`, { method: 'POST' })
        .then(res => res.json())
        .then(data => {
            currentMode = data.mode;
            document.querySelectorAll(".mode-pill").forEach(btn => btn.classList.remove("active"));
            if (mode === "RL") {
                const btn = document.getElementById("btn-mode-rl");
                if (btn) btn.classList.add("active");
            } else if (mode === "OSPF") {
                const btn = document.getElementById("btn-mode-ospf");
                if (btn) btn.classList.add("active");
            }
            if (data.telemetry) updateSimDashboard(data.telemetry);
        });
}

function changeTopology(topoName) {
    fetch(`/api/topology/${topoName}`, { method: 'POST' })
        .then(res => res.json())
        .then(data => {
            activeTopologyType = topoName;
            particles = [];
            scaledNodePositions = {};
            const rect = canvasContainer.getBoundingClientRect();
            recomputeScaledPositions(rect.width, rect.height);
        });
}

function setTrafficPreset(intensity, label) {
    burstActive = false;
    const btnBurst = document.getElementById("btn-burst");
    if (btnBurst) btnBurst.classList.remove("active");

    fetch(`/api/intensity/${intensity}`, { method: 'POST' })
        .then(res => res.json())
        .then(data => {
            const slider = document.getElementById("traffic-slider");
            const sliderVal = document.getElementById("traffic-slider-val");
            const pctVal = Math.round(intensity * 100);
            savedSliderBeforeBurst = pctVal;
            if (slider) slider.value = pctVal;
            if (sliderVal) sliderVal.textContent = label;

            document.querySelectorAll(".preset-chip").forEach(b => b.classList.remove("active"));
            if (intensity <= 0.4) {
                const p1 = document.getElementById("preset-1");
                if (p1) p1.classList.add("active");
            } else if (intensity <= 1.0) {
                const p2 = document.getElementById("preset-2");
                if (p2) p2.classList.add("active");
            } else {
                const p3 = document.getElementById("preset-3");
                if (p3) p3.classList.add("active");
            }
            if (data.telemetry) updateSimDashboard(data.telemetry);
        });
}

let sliderDebounceTimer = null;

function onTrafficSliderInput(val) {
    const sliderVal = document.getElementById("traffic-slider-val");
    if (sliderVal) sliderVal.textContent = `${val}%`;

    clearTimeout(sliderDebounceTimer);
    sliderDebounceTimer = setTimeout(() => {
        onTrafficSliderChange(val);
    }, 120);
}

function onTrafficSliderChange(val) {
    const pctVal = parseFloat(val);
    savedSliderBeforeBurst = pctVal;
    burstActive = false;
    const btnBurst = document.getElementById("btn-burst");
    if (btnBurst) btnBurst.classList.remove("active");

    const intensity = pctVal / 100.0;
    fetch(`/api/intensity/${intensity}`, { method: 'POST' })
        .then(res => res.json())
        .then(data => {
            document.querySelectorAll(".preset-chip").forEach(b => b.classList.remove("active"));
            if (data.telemetry) updateSimDashboard(data.telemetry);
        })
        .catch(err => console.error("Slider update error:", err));
}

function toggleBurst() {
    burstActive = !burstActive;
    const slider = document.getElementById("traffic-slider");
    const sliderVal = document.getElementById("traffic-slider-val");

    if (burstActive) {
        if (slider) {
            savedSliderBeforeBurst = parseInt(slider.value) || 85;
        }
        if (sliderVal) sliderVal.textContent = `⚡ ${savedSliderBeforeBurst}%+`;
    } else {
        const restoreVal = savedSliderBeforeBurst || 85;
        if (slider) slider.value = restoreVal;
        if (sliderVal) sliderVal.textContent = `${restoreVal}%`;
    }

    fetch(`/api/burst/${burstActive}`, { method: 'POST' })
        .then(res => res.json())
        .then(data => {
            const btn = document.getElementById("btn-burst");
            if (btn) {
                if (burstActive) btn.classList.add("active");
                else btn.classList.remove("active");
            }
            if (data.telemetry) updateSimDashboard(data.telemetry);
        });
}

// ================= 8. SAYFA BAŞLATMA =================
window.addEventListener("DOMContentLoaded", () => {
    resizeCanvas();
    initSimSocket();
    requestAnimationFrame(renderTopology);
});
