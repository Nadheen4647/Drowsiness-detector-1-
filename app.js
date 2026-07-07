// Application State
const state = {
    // Configs
    earThreshold: 0.20,
    marThreshold: 0.65,
    triggerDelay: 1.5, // seconds
    
    // Telemetry History (Last 100 frames)
    historyLimit: 100,
    earHistory: Array(100).fill(0.30),
    marHistory: Array(100).fill(0.15),
    
    // Live values
    ear: 0.30,
    mar: 0.15,
    fps: 30,
    
    // Monitoring state
    eyeClosedFrames: 0,
    yawningFrames: 0,
    faceLostFrames: 0,
    eyeState: 'OPEN', // OPEN, CLOSED
    mouthState: 'NORMAL', // NORMAL, YAWNING
    alarmActive: false,
    trackingAlarmActive: false,
    
    // Input Mode
    mode: 'simulate', // 'camera' or 'simulate'
    selectedScenario: 'alert', // alert, drowsy, yawning, manual
    
    // Simulator controls
    simEyeOpen: 100, // 0 to 100
    simMouthOpen: 0,  // 0 to 100
    simTimer: 0
};

// UI Elements
const badge = document.getElementById('system-badge');
const earValueEl = document.getElementById('ear-value');
const earBarEl = document.getElementById('ear-bar');
const earStatusEl = document.getElementById('ear-status');
const marValueEl = document.getElementById('mar-value');
const marBarEl = document.getElementById('mar-bar');
const marStatusEl = document.getElementById('mar-status');

const earThreshDisp = document.getElementById('ear-thresh-display');
const marThreshDisp = document.getElementById('mar-thresh-display');
const earSlider = document.getElementById('slider-ear-threshold');
const marSlider = document.getElementById('slider-mar-threshold');
const delaySlider = document.getElementById('slider-trigger-delay');

const valEarThresh = document.getElementById('val-ear-threshold');
const valMarThresh = document.getElementById('val-mar-threshold');
const valTriggerDelay = document.getElementById('val-trigger-delay');

const btnCamera = document.getElementById('btn-camera');
const btnSimulate = document.getElementById('btn-simulate');
const scenarioSelect = document.getElementById('scenario-select');
const manualControls = document.getElementById('manual-controls');
const simulatorPanel = document.getElementById('simulator-panel');
const simulatorTip = document.getElementById('simulator-tip');
const simEyeSlider = document.getElementById('sim-eye-slider');
const simMouthSlider = document.getElementById('sim-mouth-slider');

const alarmOverlay = document.getElementById('alarm-overlay');
const loadingOverlay = document.getElementById('loading-overlay');
const consoleLogs = document.getElementById('console-logs');
const btnClearLog = document.getElementById('btn-clear-log');

const hudEye = document.getElementById('hud-eye');
const hudMouth = document.getElementById('hud-mouth');
const hudFps = document.getElementById('hud-fps');

// Canvas contexts
const videoCanvas = document.getElementById('overlay-canvas');
const videoCtx = videoCanvas.getContext('2d');
const earCanvas = document.getElementById('chart-ear');
const earCtx = earCanvas.getContext('2d');
const marCanvas = document.getElementById('chart-mar');
const marCtx = marCanvas.getContext('2d');

// Webcam stream variables
const videoElement = document.getElementById('webcam');
let cameraInstance = null;
let faceMeshInstance = null;
let animationFrameId = null;
let lastFpsTime = 0;
let frameCount = 0;

// Audio context for alarms (Web Audio API)
let audioCtx = null;
let alarmIntervalId = null;

// ==========================================
// 1. Tab Switching Controller
// ==========================================
document.querySelectorAll('.tab-btn').forEach(btn => {
    btn.addEventListener('click', () => {
        document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
        document.querySelectorAll('.tab-pane').forEach(p => p.classList.remove('active'));
        
        btn.classList.add('active');
        const paneId = btn.getAttribute('data-tab');
        document.getElementById(paneId).classList.add('active');
    });
});

// ==========================================
// 2. Logger Terminal
// ==========================================
function logEvent(message, type = 'info') {
    const now = new Date();
    const timeStr = now.toTimeString().split(' ')[0];
    const logLine = document.createElement('div');
    logLine.className = `log-line text-${type}`;
    logLine.textContent = `[${timeStr}] ${message}`;
    consoleLogs.appendChild(logLine);
    consoleLogs.scrollTop = consoleLogs.scrollHeight;
}

btnClearLog.addEventListener('click', () => {
    consoleLogs.innerHTML = '';
    logEvent('Logs cleared.', 'info');
});

// ==========================================
// 3. Audio Alert System (Web Audio API)
// ==========================================
function playBeep(frequency, duration) {
    if (!audioCtx) {
        audioCtx = new (window.AudioContext || window.webkitAudioContext)();
    }
    if (audioCtx.state === 'suspended') {
        audioCtx.resume();
    }
    
    try {
        const osc = audioCtx.createOscillator();
        const gainNode = audioCtx.createGain();
        
        osc.type = 'sawtooth'; // Piercing buzzer-like waveform
        osc.frequency.setValueAtTime(frequency, audioCtx.currentTime);
        
        gainNode.gain.setValueAtTime(0.15, audioCtx.currentTime);
        gainNode.gain.exponentialRampToValueAtTime(0.001, audioCtx.currentTime + duration);
        
        osc.connect(gainNode);
        gainNode.connect(audioCtx.destination);
        
        osc.start();
        osc.stop(audioCtx.currentTime + duration);
    } catch (e) {
        console.error("Audio beep failed", e);
    }
}

/**
 * Indian Ambulance "PEE-PAW" siren:
 * HIGH tone (940→980 Hz, 450ms) followed by LOW tone (680→640 Hz, 450ms).
 * Sine waves mimic the clean electronic horn used on Indian ambulances.
 * Total one PEE-PAW cycle = 900ms; called every 900ms for seamless looping.
 */
function playIndianAmbulance() {
    if (!audioCtx) {
        audioCtx = new (window.AudioContext || window.webkitAudioContext)();
    }
    if (audioCtx.state === 'suspended') audioCtx.resume();

    try {
        const now     = audioCtx.currentTime;
        const toneDur = 0.44;  // duration of each tone (PEE or PAW)
        const gap     = 0.01;  // tiny crossfade gap between tones

        // ── PEE: HIGH tone (rises from 940 → 985 Hz) ──────────────────────
        const oscHi   = audioCtx.createOscillator();
        const gainHi  = audioCtx.createGain();
        oscHi.type = 'sine';
        oscHi.frequency.setValueAtTime(940, now);
        oscHi.frequency.linearRampToValueAtTime(985, now + toneDur);
        gainHi.gain.setValueAtTime(0.0,  now);
        gainHi.gain.linearRampToValueAtTime(0.45, now + 0.025);          // sharp attack
        gainHi.gain.setValueAtTime(0.45, now + toneDur - 0.03);
        gainHi.gain.linearRampToValueAtTime(0.0,  now + toneDur);        // fast decay
        oscHi.connect(gainHi);
        gainHi.connect(audioCtx.destination);
        oscHi.start(now);
        oscHi.stop(now + toneDur + 0.01);

        // Harmonic body for PEE (triangle at 0.5× freq — adds warmth)
        const oscHiH  = audioCtx.createOscillator();
        const gainHiH = audioCtx.createGain();
        oscHiH.type = 'triangle';
        oscHiH.frequency.setValueAtTime(470, now);
        oscHiH.frequency.linearRampToValueAtTime(492, now + toneDur);
        gainHiH.gain.setValueAtTime(0.0,  now);
        gainHiH.gain.linearRampToValueAtTime(0.10, now + 0.025);
        gainHiH.gain.setValueAtTime(0.10, now + toneDur - 0.03);
        gainHiH.gain.linearRampToValueAtTime(0.0,  now + toneDur);
        oscHiH.connect(gainHiH);
        gainHiH.connect(audioCtx.destination);
        oscHiH.start(now);
        oscHiH.stop(now + toneDur + 0.01);

        // ── PAW: LOW tone (falls from 680 → 638 Hz) ───────────────────────
        const tLow   = now + toneDur + gap;
        const oscLo  = audioCtx.createOscillator();
        const gainLo = audioCtx.createGain();
        oscLo.type = 'sine';
        oscLo.frequency.setValueAtTime(680, tLow);
        oscLo.frequency.linearRampToValueAtTime(638, tLow + toneDur);
        gainLo.gain.setValueAtTime(0.0,  tLow);
        gainLo.gain.linearRampToValueAtTime(0.45, tLow + 0.025);         // sharp attack
        gainLo.gain.setValueAtTime(0.45, tLow + toneDur - 0.03);
        gainLo.gain.linearRampToValueAtTime(0.0,  tLow + toneDur);       // fast decay
        oscLo.connect(gainLo);
        gainLo.connect(audioCtx.destination);
        oscLo.start(tLow);
        oscLo.stop(tLow + toneDur + 0.01);

        // Harmonic body for PAW
        const oscLoH  = audioCtx.createOscillator();
        const gainLoH = audioCtx.createGain();
        oscLoH.type = 'triangle';
        oscLoH.frequency.setValueAtTime(340, tLow);
        oscLoH.frequency.linearRampToValueAtTime(319, tLow + toneDur);
        gainLoH.gain.setValueAtTime(0.0,  tLow);
        gainLoH.gain.linearRampToValueAtTime(0.10, tLow + 0.025);
        gainLoH.gain.setValueAtTime(0.10, tLow + toneDur - 0.03);
        gainLoH.gain.linearRampToValueAtTime(0.0,  tLow + toneDur);
        oscLoH.connect(gainLoH);
        gainLoH.connect(audioCtx.destination);
        oscLoH.start(tLow);
        oscLoH.stop(tLow + toneDur + 0.01);

    } catch (e) {
        console.error('Indian ambulance sound failed', e);
    }
}

function startAlarm() {
    if (state.alarmActive) return;
    state.alarmActive = true;
    
    // Toggle UI flashing overlays
    alarmOverlay.style.display = 'flex';
    badge.className = 'badge-danger';
    badge.textContent = 'WARNING: DROWSY';
    
    hudEye.querySelector('.hud-indicator').className = 'hud-indicator danger';
    hudEye.querySelector('.hud-text').textContent = 'EYES: DROWSY ALERT';
    
    logEvent('WARNING: Driver drowsiness detected! Activating Indian ambulance alarm.', 'danger');
    
    // Indian ambulance PEE-PAW: one full cycle = 900ms, repeat every 900ms
    playIndianAmbulance();
    alarmIntervalId = setInterval(() => {
        playIndianAmbulance();
    }, 900);
}

function stopAlarm() {
    if (!state.alarmActive) return;
    state.alarmActive = false;
    
    alarmOverlay.style.display = 'none';
    if (alarmIntervalId) {
        clearInterval(alarmIntervalId);
        alarmIntervalId = null;
    }
    
    // Restore states
    evaluateBadgeState();
    logEvent('Status returned to normal. Alarm deactivated.', 'success');
}

let trackingIntervalId = null;

function startTrackingAlarm() {
    if (state.trackingAlarmActive) return;
    state.trackingAlarmActive = true;
    
    badge.className = 'badge-danger';
    badge.textContent = 'TRACKING LOST';
    
    logEvent('WARNING: Camera tracking lost! Face is hidden or covered.', 'danger');
    
    // Rapid 3-pip warning sound (2400Hz pips)
    trackingIntervalId = setInterval(() => {
        playBeep(2400, 0.06);
        setTimeout(() => playBeep(2400, 0.06), 100);
        setTimeout(() => playBeep(2400, 0.06), 200);
    }, 850);
}

function stopTrackingAlarm() {
    if (!state.trackingAlarmActive) return;
    state.trackingAlarmActive = false;
    
    if (trackingIntervalId) {
        clearInterval(trackingIntervalId);
        trackingIntervalId = null;
    }
    
    evaluateBadgeState();
    logEvent('Face tracking restored.', 'success');
}

function evaluateBadgeState() {
    if (state.alarmActive) return;
    
    if (state.trackingAlarmActive) {
        badge.className = 'badge-danger';
        badge.textContent = 'TRACKING LOST';
        return;
    }
    
    if (state.mouthState === 'YAWNING') {
        badge.className = 'badge-warning';
        badge.textContent = 'FATIGUED: YAWNING';
        hudMouth.querySelector('.hud-indicator').className = 'hud-indicator danger';
        hudMouth.querySelector('.hud-text').textContent = 'MOUTH: YAWN DETECTED';
    } else {
        badge.className = 'badge-safe';
        badge.textContent = 'ACTIVE / SAFE';
        hudMouth.querySelector('.hud-indicator').className = 'hud-indicator active';
        hudMouth.querySelector('.hud-text').textContent = 'MOUTH: NORMAL';
    }

    if (state.eyeState === 'OPEN') {
        hudEye.querySelector('.hud-indicator').className = 'hud-indicator active';
        hudEye.querySelector('.hud-text').textContent = 'EYES: OPEN';
    } else {
        hudEye.querySelector('.hud-indicator').className = 'hud-indicator danger';
        hudEye.querySelector('.hud-text').textContent = 'EYES: CLOSED';
    }
}

// ==========================================
// 4. Calibration Slider Updates
// ==========================================
earSlider.addEventListener('input', (e) => {
    state.earThreshold = parseFloat(e.target.value);
    valEarThresh.textContent = state.earThreshold.toFixed(2);
    earThreshDisp.textContent = state.earThreshold.toFixed(2);
});

marSlider.addEventListener('input', (e) => {
    state.marThreshold = parseFloat(e.target.value);
    valMarThresh.textContent = state.marThreshold.toFixed(2);
    marThreshDisp.textContent = state.marThreshold.toFixed(2);
});

delaySlider.addEventListener('input', (e) => {
    state.triggerDelay = parseFloat(e.target.value);
    valTriggerDelay.textContent = state.triggerDelay.toFixed(1) + 's';
});

// ==========================================
// 5. Line Chart Telemetry Draw Functions
// ==========================================
function updateHistoryArrays(newEar, newMar) {
    state.earHistory.push(newEar);
    if (state.earHistory.length > state.historyLimit) state.earHistory.shift();
    
    state.marHistory.push(newMar);
    if (state.marHistory.length > state.historyLimit) state.marHistory.shift();
}

function drawTelemetryCharts() {
    drawChart(earCanvas, earCtx, state.earHistory, state.earThreshold, true);
    drawChart(marCanvas, marCtx, state.marHistory, state.marThreshold, false);
}

function drawChart(canvas, ctx, data, threshold, isEAR) {
    const width = canvas.width = canvas.parentElement.clientWidth;
    const height = canvas.height = 110;
    
    ctx.clearRect(0, 0, width, height);
    
    // Draw Grid Lines
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.05)';
    ctx.lineWidth = 1;
    for (let i = 1; i < 4; i++) {
        const y = (height / 4) * i;
        ctx.beginPath();
        ctx.moveTo(0, y);
        ctx.lineTo(width, y);
        ctx.stroke();
    }
    
    // Draw Threshold line
    const threshY = height - (threshold * height); // Assumes value represents 0-1 scale
    ctx.strokeStyle = isEAR ? 'rgba(255, 8, 68, 0.6)' : 'rgba(245, 158, 11, 0.6)';
    ctx.lineWidth = 1.5;
    ctx.setLineDash([4, 4]);
    ctx.beginPath();
    ctx.moveTo(0, threshY);
    ctx.lineTo(width, threshY);
    ctx.stroke();
    ctx.setLineDash([]); // reset

    // Draw Data Path
    if (data.length === 0) return;
    
    ctx.beginPath();
    const step = width / (state.historyLimit - 1);
    
    for (let i = 0; i < data.length; i++) {
        const x = i * step;
        // Clamp and map value to height
        const val = Math.max(0, Math.min(1, data[i]));
        const y = height - (val * height);
        
        if (i === 0) {
            ctx.moveTo(x, y);
        } else {
            ctx.lineTo(x, y);
        }
    }
    
    ctx.strokeStyle = isEAR ? '#00f2fe' : '#ff0844';
    ctx.lineWidth = 2;
    ctx.shadowBlur = 6;
    ctx.shadowColor = isEAR ? 'rgba(0, 242, 254, 0.4)' : 'rgba(255, 8, 68, 0.4)';
    ctx.stroke();
    ctx.shadowBlur = 0; // reset glow
}

// ==========================================
// 6. Vector-based Face Mesh Simulator
// ==========================================
function updateSimulatorState() {
    state.simTimer++;
    
    if (state.selectedScenario === 'alert') {
        // Normal alert driver: EAR ~0.30 - 0.33, MAR ~0.15. Occasional blinks.
        state.simEyeOpen = 100;
        state.simMouthOpen = 5;
        
        // Periodic blink logic (every 4 seconds/120 frames, blink for 4 frames)
        const period = state.simTimer % 120;
        if (period >= 0 && period <= 4) {
            state.simEyeOpen = 10; // Blink!
        }
    }
    else if (state.selectedScenario === 'drowsy') {
        // Drowsy cycle: Normal -> Eyes getting heavy -> Closed -> Wake up alarm
        const cycle = state.simTimer % 240; // 8-second loop
        
        if (cycle < 60) {
            // Normal phase
            state.simEyeOpen = 100;
            state.simMouthOpen = 5;
        } else if (cycle >= 60 && cycle < 100) {
            // Eyes droop
            const pct = (100 - (cycle - 60) * 2);
            state.simEyeOpen = Math.max(40, pct);
        } else if (cycle >= 100 && cycle < 180) {
            // Closed eyes (micro-sleep)
            state.simEyeOpen = 5;
            state.simMouthOpen = 5;
        } else {
            // Wake up alert response
            state.simEyeOpen = 100;
            state.simMouthOpen = 10;
        }
    }
    else if (state.selectedScenario === 'yawning') {
        // Yawn cycle: Normal -> Yawn starts -> Yawn peak -> Rest
        const cycle = state.simTimer % 240;
        state.simEyeOpen = 90; // Slightly tired eyes
        
        if (cycle < 40) {
            state.simMouthOpen = 5;
        } else if (cycle >= 40 && cycle < 90) {
            // Opening mouth wide
            state.simMouthOpen = (cycle - 40) * 1.8; // climb to 90%
        } else if (cycle >= 90 && cycle < 150) {
            // Peak yawn
            state.simMouthOpen = 90;
            state.simEyeOpen = 25; // eyes squinting during yawn
        } else if (cycle >= 150 && cycle < 200) {
            // Closing mouth
            state.simMouthOpen = Math.max(5, 90 - (cycle - 150) * 1.8);
        } else {
            state.simMouthOpen = 5;
        }
    }
    else if (state.selectedScenario === 'manual') {
        // Sliders map directly
        state.simEyeOpen = parseFloat(simEyeSlider.value);
        state.simMouthOpen = parseFloat(simMouthSlider.value);
    }
    
    // Map Sim percentages to theoretical EAR/MAR ranges
    // Eyelid Open (100%) -> EAR 0.32, Closed (0%) -> EAR 0.08
    state.ear = 0.08 + (state.simEyeOpen / 100) * 0.24;
    // Mouth open (100%) -> MAR 0.78, Closed (0%) -> MAR 0.12
    state.mar = 0.12 + (state.simMouthOpen / 100) * 0.66;
    
    // Add micro-noise for organic feel
    state.ear += (Math.random() - 0.5) * 0.015;
    state.mar += (Math.random() - 0.5) * 0.015;
    state.ear = Math.max(0.02, state.ear);
    state.mar = Math.max(0.02, state.mar);
}

function drawSimulatedFace() {
    const width = videoCanvas.width = videoCanvas.parentElement.clientWidth;
    const height = videoCanvas.height = videoCanvas.parentElement.clientHeight;
    
    videoCtx.fillStyle = '#020617';
    videoCtx.fillRect(0, 0, width, height);
    
    // Coordinate centers
    const cx = width / 2;
    const cy = height / 2 - 10;
    
    // Draw Grid Lines (Futuristic Overlay)
    videoCtx.strokeStyle = 'rgba(0, 242, 254, 0.03)';
    videoCtx.lineWidth = 1;
    for (let x = 0; x < width; x += 30) {
        videoCtx.beginPath(); videoCtx.moveTo(x, 0); videoCtx.lineTo(x, height); videoCtx.stroke();
    }
    for (let y = 0; y < height; y += 30) {
        videoCtx.beginPath(); videoCtx.moveTo(0, y); videoCtx.lineTo(width, y); videoCtx.stroke();
    }
    
    // 1. Draw head contour
    videoCtx.strokeStyle = 'rgba(255,255,255,0.06)';
    videoCtx.lineWidth = 2;
    videoCtx.beginPath();
    videoCtx.ellipse(cx, cy, 110, 150, 0, 0, Math.PI * 2);
    videoCtx.stroke();
    
    // 2. Draw Nose Line
    videoCtx.strokeStyle = 'rgba(255,255,255,0.1)';
    videoCtx.beginPath();
    videoCtx.moveTo(cx, cy - 30);
    videoCtx.lineTo(cx, cy + 30);
    videoCtx.stroke();
    
    // 3. Draw Eyes (Left & Right)
    const eyeOffsetX = 45;
    const eyeOffsetY = -25;
    const eyeRadiusX = 22;
    // Map simulated open level to vertical radius
    const eyeRadiusY = 1.5 + (state.simEyeOpen / 100) * 8.5;
    
    // Left eye (represented locally)
    drawVectorEye(cx - eyeOffsetX, cy + eyeOffsetY, eyeRadiusX, eyeRadiusY);
    // Right eye
    drawVectorEye(cx + eyeOffsetX, cy + eyeOffsetY, eyeRadiusX, eyeRadiusY);
    
    // 4. Draw Mouth
    const mouthOffsetY = 55;
    const mouthRadiusX = 35 - (state.simMouthOpen / 100) * 8; // mouth gets narrower horizontally as it yawns wide
    const mouthRadiusY = 3 + (state.simMouthOpen / 100) * 32;
    
    videoCtx.strokeStyle = state.mar > state.marThreshold ? 'rgba(245, 158, 11, 0.8)' : 'rgba(0, 255, 208, 0.8)';
    videoCtx.fillStyle = state.mar > state.marThreshold ? 'rgba(245, 158, 11, 0.08)' : 'rgba(0, 255, 208, 0.03)';
    videoCtx.lineWidth = 3;
    videoCtx.beginPath();
    videoCtx.ellipse(cx, cy + mouthOffsetY, mouthRadiusX, mouthRadiusY, 0, 0, Math.PI * 2);
    videoCtx.fill();
    videoCtx.stroke();
    
    // Draw landmarks overlay indicators (points & annotations)
    videoCtx.fillStyle = 'rgba(0, 242, 254, 0.7)';
    // Draw dots at critical lip points
    const dots = [
        [cx - mouthRadiusX, cy + mouthOffsetY],
        [cx + mouthRadiusX, cy + mouthOffsetY],
        [cx, cy + mouthOffsetY - mouthRadiusY],
        [cx, cy + mouthOffsetY + mouthRadiusY]
    ];
    dots.forEach(d => {
        videoCtx.beginPath();
        videoCtx.arc(d[0], d[1], 3, 0, Math.PI * 2);
        videoCtx.fill();
    });
    
    // Draw Text annotations on target regions
    videoCtx.font = '10px Outfit';
    videoCtx.fillStyle = 'rgba(255,255,255,0.4)';
    videoCtx.fillText(`EAR: ${state.ear.toFixed(2)}`, cx - 75, cy + eyeOffsetY - 25);
    videoCtx.fillText(`MAR: ${state.mar.toFixed(2)}`, cx - 25, cy + mouthOffsetY + mouthRadiusY + 18);
    
    // HUD overlay text
    videoCtx.font = '12px Outfit';
    videoCtx.fillStyle = 'rgba(0, 242, 254, 0.8)';
    videoCtx.fillText("[LANDMARK MODE: SIMULATOR VECTORS]", 15, 25);
}

function drawVectorEye(ex, ey, rx, ry) {
    // Eye background
    videoCtx.fillStyle = 'rgba(255, 255, 255, 0.03)';
    videoCtx.beginPath();
    videoCtx.ellipse(ex, ey, rx, ry, 0, 0, Math.PI * 2);
    videoCtx.fill();

    // Eyelid line
    videoCtx.strokeStyle = state.ear < state.earThreshold ? 'rgba(255, 8, 68, 0.9)' : 'rgba(0, 242, 254, 0.8)';
    videoCtx.lineWidth = 2.5;
    videoCtx.beginPath();
    videoCtx.ellipse(ex, ey, rx, ry, 0, 0, Math.PI * 2);
    videoCtx.stroke();

    // Iris (only draw if eyes open enough)
    if (ry > 3) {
        videoCtx.fillStyle = '#02c39a';
        videoCtx.beginPath();
        videoCtx.arc(ex, ey, Math.min(ry - 1, 6), 0, Math.PI * 2);
        videoCtx.fill();
        
        videoCtx.fillStyle = '#000000';
        videoCtx.beginPath();
        videoCtx.arc(ex, ey, Math.min(ry - 2, 2.5), 0, Math.PI * 2);
        videoCtx.fill();
    }
    
    // Draw localized measurement helper lines
    videoCtx.strokeStyle = 'rgba(255, 255, 255, 0.15)';
    videoCtx.lineWidth = 1;
    // vertical
    videoCtx.beginPath();
    videoCtx.moveTo(ex, ey - ry);
    videoCtx.lineTo(ex, ey + ry);
    videoCtx.stroke();
    // horizontal
    videoCtx.beginPath();
    videoCtx.moveTo(ex - rx, ey);
    videoCtx.lineTo(ex + rx, ey);
    videoCtx.stroke();
}

// ==========================================
// 7. Core Drowsiness Evaluation Algorithm
// ==========================================
function evaluateDrowsiness() {
    const fps = state.fps || 30;
    const requiredClosedFrames = state.triggerDelay * fps;
    
    // Evaluate EAR
    if (state.ear < state.earThreshold) {
        state.eyeClosedFrames++;
        state.eyeState = 'CLOSED';
        earStatusEl.className = 'status-text-danger';
        earStatusEl.textContent = `Closed (${Math.round((state.eyeClosedFrames / fps) * 10) / 10}s)`;
        
        if (state.eyeClosedFrames === 5) {
            logEvent('Notice: Driver eyelids closing.', 'warning');
        }
        
        if (state.eyeClosedFrames >= requiredClosedFrames) {
            startAlarm();
        }
    } else {
        if (state.eyeClosedFrames > 0 && !state.alarmActive) {
            logEvent(`Event: Eyelids opened after ${Math.round((state.eyeClosedFrames / fps) * 10) / 10} seconds.`, 'info');
        }
        state.eyeClosedFrames = 0;
        state.eyeState = 'OPEN';
        earStatusEl.className = 'status-text-safe';
        earStatusEl.textContent = 'Normal';
        
        // Stop the alarm if eyes are fully open
        if (state.ear > state.earThreshold + 0.02) {
            stopAlarm();
        }
    }
    
    // Evaluate MAR
    if (state.mar > state.marThreshold) {
        state.yawningFrames++;
        
        if (state.yawningFrames === 15) {
            state.mouthState = 'YAWNING';
            evaluateBadgeState();
            logEvent('ALERT: Driver yawning detected! Fatigue warnings engaged.', 'warning');
        }
        
        marStatusEl.className = 'status-text-warning';
        marStatusEl.textContent = `Yawning (${Math.round((state.yawningFrames / fps) * 10) / 10}s)`;
    } else {
        if (state.yawningFrames > 0) {
            state.mouthState = 'NORMAL';
            evaluateBadgeState();
        }
        state.yawningFrames = 0;
        marStatusEl.className = 'status-text-safe';
        marStatusEl.textContent = 'Resting';
    }
}

// ==========================================
// 8. Live Stats Elements Sync
// ==========================================
function updateDashboardMetrics() {
    earValueEl.textContent = state.ear.toFixed(2);
    // map 0.0-0.4 to 0-100% progress
    const earBarPct = Math.min(100, Math.max(0, (state.ear / 0.4) * 100));
    earBarEl.style.width = `${earBarPct}%`;
    
    if (state.ear < state.earThreshold) {
        earBarEl.className = 'progress-bar progress-danger';
    } else {
        earBarEl.className = 'progress-bar progress-safe';
    }
    
    marValueEl.textContent = state.mar.toFixed(2);
    // map 0.0-1.0 to 0-100% progress
    const marBarPct = Math.min(100, Math.max(0, state.mar * 100));
    marBarEl.style.width = `${marBarPct}%`;
    
    if (state.mar > state.marThreshold) {
        marBarEl.className = 'progress-bar progress-warning';
    } else {
        marBarEl.className = 'progress-bar progress-safe';
    }
}

// ==========================================
// 9. Main Application Loops
// ==========================================
function simulatorLoop(timestamp) {
    if (state.mode !== 'simulate') return;
    
    // Calculate FPS
    if (!lastFpsTime) lastFpsTime = timestamp;
    frameCount++;
    if (timestamp > lastFpsTime + 1000) {
        state.fps = Math.round((frameCount * 1000) / (timestamp - lastFpsTime));
        hudFps.textContent = `FPS: ${state.fps}`;
        lastFpsTime = timestamp;
        frameCount = 0;
    }
    
    updateSimulatorState();
    evaluateDrowsiness();
    updateDashboardMetrics();
    updateHistoryArrays(state.ear, state.mar);
    drawTelemetryCharts();
    drawSimulatedFace();
    
    animationFrameId = requestAnimationFrame(simulatorLoop);
}

// ==========================================
// 10. Web Camera & MediaPipe Integration
// ==========================================
function getEuclideanDistance(pt1, pt2, width, height) {
    const dx = (pt1.x - pt2.x) * width;
    const dy = (pt1.y - pt2.y) * height;
    return Math.sqrt(dx * dx + dy * dy);
}

function processWebcamResults(results) {
    if (state.mode !== 'camera') return;
    
    // Calculate FPS
    const timestamp = performance.now();
    if (!lastFpsTime) lastFpsTime = timestamp;
    frameCount++;
    if (timestamp > lastFpsTime + 1000) {
        state.fps = Math.round((frameCount * 1000) / (timestamp - lastFpsTime));
        hudFps.textContent = `FPS: ${state.fps}`;
        lastFpsTime = timestamp;
        frameCount = 0;
    }
    
    const width = videoCanvas.width = videoCanvas.parentElement.clientWidth;
    const height = videoCanvas.height = videoCanvas.parentElement.clientHeight;
    
    // Draw camera frame on canvas mirror-flipped
    videoCtx.save();
    videoCtx.translate(width, 0);
    videoCtx.scale(-1, 1);
    videoCtx.drawImage(results.image, 0, 0, width, height);
    videoCtx.restore();
    
    let faceFound = false;
    
    if (results.multiFaceLandmarks && results.multiFaceLandmarks.length > 0) {
        faceFound = true;
        const landmarks = results.multiFaceLandmarks[0];
        
        // MediaPipe Face Mesh Index Mappings:
        // Left Eye: Inner [133], Outer [33], Top-Inner [160], Top-Outer [158], Bottom-Inner [144], Bottom-Outer [153]
        const leftEyePoints = [landmarks[33], landmarks[160], landmarks[158], landmarks[133], landmarks[153], landmarks[144]];
        
        // Right Eye: Inner [362], Outer [263], Top-Inner [385], Top-Outer [387], Bottom-Inner [380], Bottom-Outer [373]
        const rightEyePoints = [landmarks[263], landmarks[385], landmarks[387], landmarks[362], landmarks[373], landmarks[380]];
        
        // Calculate EAR for left and right eye
        const earLeft = calculateEyeRatio(leftEyePoints, width, height);
        const earRight = calculateEyeRatio(rightEyePoints, width, height);
        state.ear = (earLeft + earRight) / 2;
        
        // Mouth outer contours: Left [61], Right [291], Verticals [37, 0, 267] vs [84, 17, 314]
        const p61 = landmarks[61];
        const p291 = landmarks[291];
        const p37 = landmarks[37];
        const p84 = landmarks[84];
        const p0 = landmarks[0];
        const p17 = landmarks[17];
        const p267 = landmarks[267];
        const p314 = landmarks[314];
        
        const dVertical1 = getEuclideanDistance(p37, p84, width, height);
        const dVertical2 = getEuclideanDistance(p0, p17, width, height);
        const dVertical3 = getEuclideanDistance(p267, p314, width, height);
        const dHorizontal = getEuclideanDistance(p61, p291, width, height);
        
        state.mar = (dVertical1 + dVertical2 + dVertical3) / (2 * dHorizontal);
        
        state.faceLostFrames = 0;
        stopTrackingAlarm();
        
        // Draw Facial Landmarks Mesh
        drawMeshLandmarks(landmarks, width, height);
    }
    
    if (!faceFound) {
        state.faceLostFrames++;
        if (state.faceLostFrames >= 30) {
            startTrackingAlarm();
        }
        
        // Draw Warning Face lost
        videoCtx.fillStyle = 'rgba(255, 8, 68, 0.15)';
        videoCtx.fillRect(0, 0, width, height);
        
        videoCtx.font = 'bold 20px Outfit';
        videoCtx.fillStyle = '#ff0844';
        videoCtx.textAlign = 'center';
        videoCtx.fillText("TRACKING LOST - NO FACE DETECTED", width / 2, height / 2);
        videoCtx.textAlign = 'left'; // reset
        
        // Reset states temporarily to avoid latching
        state.ear = 0.32;
        state.mar = 0.15;
    }
    
    evaluateDrowsiness();
    updateDashboardMetrics();
    updateHistoryArrays(state.ear, state.mar);
    drawTelemetryCharts();
}

function calculateEyeRatio(pts, w, h) {
    const dVert1 = getEuclideanDistance(pts[1], pts[5], w, h);
    const dVert2 = getEuclideanDistance(pts[2], pts[4], w, h);
    const dHoriz = getEuclideanDistance(pts[0], pts[3], w, h);
    return (dVert1 + dVert2) / (2 * dHoriz);
}

function drawMeshLandmarks(landmarks, w, h) {
    videoCtx.fillStyle = 'rgba(0, 255, 208, 0.45)';
    videoCtx.strokeStyle = 'rgba(0, 242, 254, 0.2)';
    videoCtx.lineWidth = 0.5;
    
    // Draw all 468 points on the face
    for (let i = 0; i < landmarks.length; i++) {
        // Mirror landmarker points to align with flipped webcam
        const x = (1 - landmarks[i].x) * w;
        const y = landmarks[i].y * h;
        
        videoCtx.beginPath();
        videoCtx.arc(x, y, 1.2, 0, Math.PI * 2);
        videoCtx.fill();
    }
    
    // Outline critical features: Left eye, Right eye, Mouth
    drawFeatureBoundary([33, 160, 158, 133, 153, 144], landmarks, w, h, 'rgba(0, 242, 254, 0.8)');
    drawFeatureBoundary([263, 385, 387, 362, 373, 380], landmarks, w, h, 'rgba(0, 242, 254, 0.8)');
    drawFeatureBoundary([61, 37, 0, 267, 291, 314, 17, 84], landmarks, w, h, 'rgba(0, 255, 208, 0.8)');
}

function drawFeatureBoundary(indices, landmarks, w, h, color) {
    videoCtx.strokeStyle = color;
    videoCtx.lineWidth = 1.5;
    videoCtx.beginPath();
    
    for (let i = 0; i < indices.length; i++) {
        const pt = landmarks[indices[i]];
        const x = (1 - pt.x) * w;
        const y = pt.y * h;
        
        if (i === 0) videoCtx.moveTo(x, y);
        else videoCtx.lineTo(x, y);
    }
    videoCtx.closePath();
    videoCtx.stroke();
}

// ==========================================
// 11. Event Bindings: Camera vs Simulator
// ==========================================
async function initWebcam() {
    loadingOverlay.style.display = 'flex';
    logEvent('Requesting web camera access...', 'info');
    
    // Check if running on local file system (file:// protocol) which blocks media devices
    if (window.location.protocol === 'file:') {
        loadingOverlay.style.display = 'none';
        const msg = 'Webcam Error: Browsers block camera access on local files (file://). ' +
                    'Please run a local web server (e.g. run "python -m http.server 8000" in your terminal) ' +
                    'and open http://localhost:8000/project.html to use the camera.';
        logEvent(msg, 'danger');
        alert(msg);
        switchToSimulator();
        return;
    }
    
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
        loadingOverlay.style.display = 'none';
        const msg = 'Webcam Error: Camera access API (navigator.mediaDevices.getUserMedia) is not supported by your browser or requires a secure context (HTTPS or localhost).';
        logEvent(msg, 'danger');
        alert(msg);
        switchToSimulator();
        return;
    }
    
    try {
        const constraints = { video: { width: 640, height: 480, facingMode: 'user' } };
        const stream = await navigator.mediaDevices.getUserMedia(constraints);
        videoElement.srcObject = stream;
        await videoElement.play();
        logEvent('Camera stream obtained. Initializing MediaPipe FaceMesh...', 'info');
        
        // Initialize MediaPipe FaceMesh
        faceMeshInstance = new FaceMesh({
            locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/face_mesh/${file}`
        });
        
        faceMeshInstance.setOptions({
            maxNumFaces: 1,
            refineLandmarks: false, // keep it lightweight for high FPS
            minDetectionConfidence: 0.5,
            minTrackingConfidence: 0.5
        });
        
        faceMeshInstance.onResults(processWebcamResults);
        
        // Setup MediaPipe Camera utility
        cameraInstance = new Camera(videoElement, {
            onFrame: async () => {
                if (state.mode === 'camera') {
                    await faceMeshInstance.send({ image: videoElement });
                }
            },
            width: 640,
            height: 480
        });
        
        await cameraInstance.start();
        
        loadingOverlay.style.display = 'none';
        logEvent('MediaPipe FaceMesh loaded successfully. Live AI monitoring active.', 'success');
        
        state.mode = 'camera';
        btnCamera.classList.add('btn-active');
        btnSimulate.classList.remove('btn-active');
        simulatorPanel.style.display = 'none';
        
    } catch (err) {
        loadingOverlay.style.display = 'none';
        logEvent(`Webcam initialization failed: ${err.message}. Reverting to Simulator.`, 'danger');
        alert(`Webcam Error: ${err.message}. The application will run in simulation mode.`);
        switchToSimulator();
    }
}

function stopWebcam() {
    if (cameraInstance) {
        try {
            cameraInstance.stop();
        } catch (e) {}
        cameraInstance = null;
    }
    if (videoElement.srcObject) {
        videoElement.srcObject.getTracks().forEach(track => track.stop());
        videoElement.srcObject = null;
    }
    if (faceMeshInstance) {
        faceMeshInstance.close();
        faceMeshInstance = null;
    }
    logEvent('Camera stream terminated.', 'info');
}

function switchToSimulator() {
    stopWebcam();
    state.mode = 'simulate';
    btnCamera.classList.remove('btn-active');
    btnSimulate.classList.add('btn-active');
    simulatorPanel.style.display = 'flex';
    
    // Restart animation loop
    if (animationFrameId) cancelAnimationFrame(animationFrameId);
    animationFrameId = requestAnimationFrame(simulatorLoop);
    logEvent('Switched to Interactive Simulator mode.', 'info');
}

// Action Button Listeners
btnCamera.addEventListener('click', () => {
    if (state.mode === 'camera') return; // already running
    initWebcam();
});

btnSimulate.addEventListener('click', () => {
    if (state.mode === 'simulate') return; // already running
    switchToSimulator();
});

// Scenario updates
scenarioSelect.addEventListener('change', (e) => {
    state.selectedScenario = e.target.value;
    logEvent(`Simulator preset changed to: "${e.target.value.toUpperCase()}"`, 'info');
    
    if (state.selectedScenario === 'manual') {
        manualControls.style.display = 'flex';
        simulatorTip.textContent = 'Use the sliders below to manually control the driver\'s eyes and mouth levels!';
    } else {
        manualControls.style.display = 'none';
        simulatorTip.textContent = 'Running auto-simulation. Switch preset or select "Manual Overrides" to control face features manually!';
    }
    
    // Stop alarms when shifting presets to avoid latching alert
    stopAlarm();
});

// ==========================================
// 12. Application Booting
// ==========================================
window.addEventListener('load', () => {
    // Initial draw to clean canvases
    drawTelemetryCharts();
    
    // Start Simulator by default
    animationFrameId = requestAnimationFrame(simulatorLoop);
});
