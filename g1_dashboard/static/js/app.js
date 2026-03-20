/**
 * G1 Biped Robot RL Dashboard — Frontend Logic
 * =============================================
 * - Fetches training status from FastAPI backend
 * - Updates KPI cards and Plotly charts in real-time via WebSocket
 * - Supports one-click model inference with joint trajectory visualization
 */

// ── Config ──
const API_BASE = '';  // Same origin
const POLL_INTERVAL = 3000;  // 3 seconds
const TARGET_TIMESTEPS = 10_000_000;

// ── State ──
let rewardHistory = { x: [], y: [] };
let klHistory = { x: [], y: [] };
let ws = null;

// ═══════════════════════════════════════════
// Initialization
// ═══════════════════════════════════════════

document.addEventListener('DOMContentLoaded', () => {
    initCharts();
    fetchStatus();
    fetchExperiments();
    fetchSystemInfo();
    connectWebSocket();
    
    // Fallback polling if WebSocket fails
    setInterval(fetchStatus, POLL_INTERVAL);
});


// ═══════════════════════════════════════════
// KPI Updates
// ═══════════════════════════════════════════

async function fetchStatus() {
    try {
        const res = await fetch(`${API_BASE}/api/status`);
        const data = await res.json();
        updateKPIs(data);
        
        // Update badge
        const badge = document.getElementById('training-badge');
        const statusText = document.getElementById('training-status-text');
        if (data.is_training) {
            badge.classList.remove('offline');
            statusText.textContent = 'Training Active';
        } else {
            badge.classList.add('offline');
            statusText.textContent = 'Idle';
        }
    } catch (e) {
        console.warn('Status fetch failed:', e);
    }
}

function updateKPIs(data) {
    const tsEl = document.getElementById('kpi-timesteps');
    const rewEl = document.getElementById('kpi-reward');
    const fpsEl = document.getElementById('kpi-fps');
    const klEl = document.getElementById('kpi-kl');
    const progressFill = document.getElementById('progress-fill');
    
    if (data.total_timesteps) {
        const ts = Number(data.total_timesteps);
        tsEl.textContent = ts >= 1_000_000 
            ? (ts / 1_000_000).toFixed(2) + 'M'
            : ts >= 1_000 
                ? (ts / 1_000).toFixed(0) + 'K'
                : ts;
        
        const pct = Math.min((ts / TARGET_TIMESTEPS) * 100, 100);
        progressFill.style.width = pct + '%';
    }
    
    if (data.ep_rew_mean !== undefined) {
        rewEl.textContent = Number(data.ep_rew_mean).toFixed(1);
        rewEl.style.color = data.ep_rew_mean > -500 ? '#10b981' : '#ef4444';
    }
    
    if (data.fps) {
        fpsEl.textContent = Number(data.fps).toFixed(0);
    }
    
    if (data.approx_kl !== undefined) {
        klEl.textContent = Number(data.approx_kl).toFixed(5);
        klEl.style.color = data.approx_kl < 0.02 ? '#10b981' : '#f59e0b';
    }
}


// ═══════════════════════════════════════════
// Plotly Charts
// ═══════════════════════════════════════════

const CHART_LAYOUT = {
    paper_bgcolor: 'transparent',
    plot_bgcolor: 'transparent',
    font: { family: 'Inter, sans-serif', color: '#9ca3af', size: 11 },
    margin: { l: 50, r: 20, t: 10, b: 40 },
    xaxis: { gridcolor: '#1f2937', zerolinecolor: '#374151' },
    yaxis: { gridcolor: '#1f2937', zerolinecolor: '#374151' },
    showlegend: false,
};

function initCharts() {
    // Reward chart
    Plotly.newPlot('reward-chart', [{
        x: [], y: [],
        type: 'scatter',
        mode: 'lines',
        line: { color: '#22d3ee', width: 2, shape: 'spline' },
        fill: 'tozeroy',
        fillcolor: 'rgba(34, 211, 238, 0.08)',
        name: 'Reward',
    }], {
        ...CHART_LAYOUT,
        xaxis: { ...CHART_LAYOUT.xaxis, title: 'Timesteps' },
        yaxis: { ...CHART_LAYOUT.yaxis, title: 'Ep Reward Mean' },
    }, { responsive: true, displayModeBar: false });
    
    // KL chart
    Plotly.newPlot('kl-chart', [{
        x: [], y: [],
        type: 'scatter',
        mode: 'lines',
        line: { color: '#a855f7', width: 2, shape: 'spline' },
        fill: 'tozeroy',
        fillcolor: 'rgba(168, 85, 247, 0.08)',
        name: 'KL Divergence',
    }], {
        ...CHART_LAYOUT,
        xaxis: { ...CHART_LAYOUT.xaxis, title: 'Timesteps' },
        yaxis: { ...CHART_LAYOUT.yaxis, title: 'Approx KL' },
    }, { responsive: true, displayModeBar: false });
    
    // Joint trajectory (empty placeholder)
    Plotly.newPlot('joint-chart', [], {
        ...CHART_LAYOUT,
        xaxis: { ...CHART_LAYOUT.xaxis, title: 'Step' },
        yaxis: { ...CHART_LAYOUT.yaxis, title: 'Joint Angle (rad)' },
        annotations: [{
            text: '点击 "Run Inference" 查看推理结果',
            xref: 'paper', yref: 'paper',
            x: 0.5, y: 0.5,
            showarrow: false,
            font: { size: 14, color: '#6b7280' },
        }],
    }, { responsive: true, displayModeBar: false });
    
    // Reward Function Breakdown
    Plotly.newPlot('reward-breakdown-chart', [{
        values: [50, 30, 20],
        labels: ['Imitation Tracking', 'Forward Velocity', 'Survival'],
        type: 'pie',
        textinfo: 'label+percent',
        hole: 0.6,
        marker: {
            colors: ['#22d3ee', '#10b981', '#a855f7']
        }
    }], {
        ...CHART_LAYOUT,
        showlegend: false,
        margin: { l: 20, r: 20, t: 20, b: 20 }
    }, { responsive: true, displayModeBar: false });
    
    // Fetch historical data
    fetchTrainingHistory();
}

async function fetchTrainingHistory() {
    try {
        const res = await fetch(`${API_BASE}/api/training-history`);
        const data = await res.json();
        
        if (data.timesteps && data.timesteps.length > 0) {
            Plotly.update('reward-chart', {
                x: [data.timesteps],
                y: [data.rewards],
            });
            
            Plotly.update('kl-chart', {
                x: [data.timesteps],
                y: [data.kl],
            });
            
            // Save for live appending
            rewardHistory.x = data.timesteps;
            rewardHistory.y = data.rewards;
            klHistory.x = data.timesteps;
            klHistory.y = data.kl;
        }
    } catch (e) {
        console.warn('History fetch failed:', e);
    }
}


// ═══════════════════════════════════════════
// WebSocket Live Stream
// ═══════════════════════════════════════════

function connectWebSocket() {
    const protocol = location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsUrl = `${protocol}//${location.host}/ws/live-training`;
    
    ws = new WebSocket(wsUrl);
    
    ws.onmessage = (event) => {
        const data = JSON.parse(event.data);
        updateKPIs(data);
        
        // Append to charts
        if (data.total_timesteps && data.ep_rew_mean !== undefined) {
            rewardHistory.x.push(data.total_timesteps);
            rewardHistory.y.push(data.ep_rew_mean);
            
            Plotly.update('reward-chart', {
                x: [rewardHistory.x],
                y: [rewardHistory.y],
            });
        }
        
        if (data.total_timesteps && data.approx_kl !== undefined) {
            klHistory.x.push(data.total_timesteps);
            klHistory.y.push(data.approx_kl);
            
            Plotly.update('kl-chart', {
                x: [klHistory.x],
                y: [klHistory.y],
            });
        }
    };
    
    ws.onclose = () => {
        console.log('WebSocket closed, reconnecting in 5s...');
        setTimeout(connectWebSocket, 5000);
    };
    
    ws.onerror = (e) => {
        console.warn('WebSocket error:', e);
    };
}


// ═══════════════════════════════════════════
// Inference
// ═══════════════════════════════════════════

async function runInference() {
    const btn = document.getElementById('btn-run-inference');
    const statusEl = document.getElementById('inference-status');
    const modelName = document.getElementById('model-select').value;
    
    btn.disabled = true;
    btn.textContent = '⏳ Running...';
    statusEl.textContent = `Loading ${modelName} and running 500-step inference...`;
    
    try {
        const res = await fetch(`${API_BASE}/api/inference?model_name=${modelName}`, {
            method: 'POST',
        });
        const data = await res.json();
        
        if (data.error) {
            statusEl.textContent = `❌ Error: ${data.error}`;
            return;
        }
        
        statusEl.textContent = `✅ Done! Steps: ${data.total_steps} | Reward: ${data.total_reward.toFixed(1)} | Ended: ${data.terminated_reason}`;
        
        // Plot joint trajectories
        const jointNames = ['L_Hip_P', 'L_Hip_R', 'L_Hip_Y', 'L_Knee', 'L_Ankle_P', 'L_Ankle_R',
                           'R_Hip_P', 'R_Hip_R', 'R_Hip_Y', 'R_Knee', 'R_Ankle_P', 'R_Ankle_R'];
        const colors = ['#22d3ee', '#3b82f6', '#a855f7', '#10b981', '#f59e0b', '#ef4444',
                        '#06b6d4', '#6366f1', '#c084fc', '#34d399', '#fbbf24', '#f87171'];
        
        const traces = [];
        for (let j = 0; j < Math.min(6, data.joint_positions[0].length); j++) {
            traces.push({
                x: data.timesteps,
                y: data.joint_positions.map(jp => jp[j]),
                type: 'scatter',
                mode: 'lines',
                name: jointNames[j],
                line: { color: colors[j], width: 1.5 },
            });
        }
        
        Plotly.react('joint-chart', traces, {
            ...CHART_LAYOUT,
            showlegend: true,
            legend: { font: { size: 10 }, orientation: 'h', y: -0.2 },
            xaxis: { ...CHART_LAYOUT.xaxis, title: 'Step' },
            yaxis: { ...CHART_LAYOUT.yaxis, title: 'Joint Angle (rad)' },
        });
        
    } catch (e) {
        statusEl.textContent = `❌ Network error: ${e.message}`;
    } finally {
        btn.disabled = false;
        btn.textContent = '▶ Run Inference';
    }
}


// ═══════════════════════════════════════════
// Experiments Table
// ═══════════════════════════════════════════

async function fetchExperiments() {
    try {
        const res = await fetch(`${API_BASE}/api/experiments`);
        const data = await res.json();
        
        const tbody = document.getElementById('experiments-body');
        if (data.experiments && data.experiments.length > 0) {
            tbody.innerHTML = data.experiments.map(exp => `
                <tr>
                    <td>${exp.name}</td>
                    <td>${exp.size_mb}</td>
                    <td>${exp.modified}</td>
                    <td><button class="btn-small" onclick="document.getElementById('model-select').value='${exp.name}';runInference();">Inference</button></td>
                </tr>
            `).join('');
        } else {
            tbody.innerHTML = '<tr><td colspan="4" style="color:#6b7280">No saved models found</td></tr>';
        }
    } catch (e) {
        console.warn('Experiments fetch failed:', e);
    }
}


// ═══════════════════════════════════════════
// System Info
// ═══════════════════════════════════════════

async function fetchSystemInfo() {
    try {
        const res = await fetch(`${API_BASE}/api/system-info`);
        const data = await res.json();
        
        document.getElementById('sys-platform').textContent = data.platform;
        document.getElementById('sys-python').textContent = data.python;
        document.getElementById('sys-hostname').textContent = data.hostname;
        
        const setLib = (id, libName) => {
            const el = document.getElementById(id);
            const val = data.libs[libName];
            if (val) {
                el.textContent = val;
                el.style.color = '#10b981';
            } else {
                el.textContent = 'Missing';
                el.style.color = '#ef4444';
            }
        };
        
        setLib('sys-sb3', 'stable_baselines3');
        setLib('sys-mujoco', 'mujoco');
        setLib('sys-torch', 'torch');
        setLib('sys-numpy', 'numpy');
        
        document.getElementById('sys-models').textContent = data.models_available.length + ' saved';
        
    } catch (e) {
        console.warn('System info fetch failed:', e);
    }
}
