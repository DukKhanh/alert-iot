// ── Canvas Setup ──
const waveCanvas = document.getElementById('waveCanvas');
const fftCanvas  = document.getElementById('fftCanvas');
const wCtx = waveCanvas.getContext('2d');
const fCtx  = fftCanvas.getContext('2d');

const COLORS = {
  bg: '#0a0c10', bg2: '#0d1014', bg3: '#11141a',
  grid: '#1a1f2a', green: '#00e57d', red: '#ff3b3b',
  yellow: '#ffd94a', cyan: '#00d4ff', text2: '#6b7a94', blue: '#3d9bff'
};

const SOUND_LABELS = ['Môi trường', 'Tiếng người', 'Máy móc', 'Tiếng nổ/va đập', 'Báo động'];
const ANOMALY_LABELS = ['Tiếng nổ/va đập', 'Báo động'];

// ── State ──
let waveData = new Float32Array(512).fill(0);
let fftData  = new Float32Array(64).fill(0);
let simActive    = false;
let anomalyMode  = false;
let anomalyTimer = 0;
let frameCount   = 0;
let anomalyTypeIdx = 3;
let relayStates  = { 1: true, 2: true, 3: false, 4: true };

// ── Canvas resize ──
function resizeCanvases() {
  waveCanvas.width  = waveCanvas.offsetWidth;
  waveCanvas.height = waveCanvas.offsetHeight;
  fftCanvas.width   = fftCanvas.offsetWidth;
  fftCanvas.height  = fftCanvas.offsetHeight;
}
window.addEventListener('resize', resizeCanvases);

// ── Clock ──
function updateClock() {
  document.getElementById('clockDisplay').textContent =
    new Date().toLocaleTimeString('vi-VN', { hour12: false });
}
setInterval(updateClock, 1000);
updateClock();

// ── Log ──
const logLines = [];

function addLog(msg, type = '') {
  const t = new Date().toLocaleTimeString('vi-VN', { hour12: false });
  logLines.push({ msg: `[${t}] ${msg}`, type });
  if (logLines.length > 100) logLines.shift();
  renderLog();
}

function renderLog() {
  const box = document.getElementById('logBox');
  box.innerHTML = logLines.slice(-30).reverse().map(l =>
    `<div class="log-line ${l.type ? 'log-' + l.type : ''}">${l.msg}</div>`
  ).join('');
}

function clearLog() {
  logLines.length = 0;
  renderLog();
}

// ── Relay toggle ──
function toggleRelay(n) {
  relayStates[n] = !relayStates[n];
  const btn = document.getElementById('relay' + n);
  btn.classList.toggle('on',  relayStates[n]);
  btn.classList.toggle('off', !relayStates[n]);
  addLog(`Relay ${n} (GPIO${24 + n}): ${relayStates[n] ? 'BẬT' : 'TẮT'}`,
         relayStates[n] ? 'ok' : 'warn');
}

// ── Alert UI ──
function setAlert(level, title, desc) {
  const box  = document.getElementById('alertBox');
  const icon = document.getElementById('alertIcon');
  const ttl  = document.getElementById('alertTitle');
  const d    = document.getElementById('alertDesc');
  const tm   = document.getElementById('alertTime');

  box.className  = 'alert-box ' + level;
  icon.className = 'alert-icon ' + level;
  ttl.className  = 'alert-title ' + level;
  icon.textContent = level === 'normal' ? '✓' : level === 'warning' ? '!' : '✕';
  ttl.textContent  = title;
  d.textContent    = desc;
  tm.textContent   = new Date().toLocaleTimeString('vi-VN', { hour12: false });
}

// ── Sound tags ──
function updateSoundTags(confs, anomalyIdx) {
  SOUND_LABELS.forEach((label, i) => {
    const tag  = document.getElementById('tag' + i);
    const fill = document.getElementById('tagFill' + i);
    const conf = document.getElementById('tagConf' + i);
    const pct  = Math.round(confs[i] * 100);

    fill.style.width   = pct + '%';
    conf.textContent   = pct + '%';
    tag.className      = 'sound-tag';

    if (i === anomalyIdx && pct > 50) {
      tag.classList.add(ANOMALY_LABELS.includes(label) ? 'danger-tag' : 'active');
    } else if (i === anomalyIdx) {
      tag.classList.add('active');
    }

    fill.style.background = (ANOMALY_LABELS.includes(label) && pct > 50)
      ? COLORS.red : COLORS.cyan;
  });
}

// ── Draw waveform ──
function drawWaveform() {
  const W = waveCanvas.width, H = waveCanvas.height;
  wCtx.clearRect(0, 0, W, H);
  wCtx.fillStyle = COLORS.bg;
  wCtx.fillRect(0, 0, W, H);

  // Grid vertical
  wCtx.strokeStyle = COLORS.grid;
  wCtx.lineWidth = 0.5;
  for (let x = 0; x < W; x += W / 8) {
    wCtx.beginPath(); wCtx.moveTo(x, 0); wCtx.lineTo(x, H); wCtx.stroke();
  }
  // Grid horizontal
  for (let y = 0; y < H; y += H / 4) {
    wCtx.beginPath(); wCtx.moveTo(0, y); wCtx.lineTo(W, y); wCtx.stroke();
  }
  // Center line
  wCtx.strokeStyle = '#1f2635';
  wCtx.lineWidth = 1;
  wCtx.beginPath(); wCtx.moveTo(0, H / 2); wCtx.lineTo(W, H / 2); wCtx.stroke();

  // Waveform line
  const color = anomalyMode ? COLORS.red : COLORS.green;
  wCtx.strokeStyle   = color;
  wCtx.lineWidth     = 1.5;
  wCtx.shadowBlur    = anomalyMode ? 6 : 3;
  wCtx.shadowColor   = color;
  wCtx.beginPath();
  for (let i = 0; i < waveData.length; i++) {
    const x = (i / waveData.length) * W;
    const y = H / 2 + waveData[i] * (H / 2 - 8);
    i === 0 ? wCtx.moveTo(x, y) : wCtx.lineTo(x, y);
  }
  wCtx.stroke();
  wCtx.shadowBlur = 0;
}

// ── Draw FFT ──
function drawFFT() {
  const W = fftCanvas.width, H = fftCanvas.height;
  fCtx.clearRect(0, 0, W, H);
  fCtx.fillStyle = COLORS.bg;
  fCtx.fillRect(0, 0, W, H);

  // Grid
  fCtx.strokeStyle = COLORS.grid;
  fCtx.lineWidth = 0.5;
  for (let y = 0; y < H; y += H / 4) {
    fCtx.beginPath(); fCtx.moveTo(0, y); fCtx.lineTo(W, y); fCtx.stroke();
  }

  // Bars
  const bars = fftData.length;
  const bw = W / bars;
  for (let i = 0; i < bars; i++) {
    const h   = fftData[i] * (H - 4);
    const hue = anomalyMode
      ? `hsl(${(i / bars) * 30}, 90%, 50%)`
      : `hsl(${170 + (i / bars) * 60}, 80%, 50%)`;
    fCtx.fillStyle = hue;
    fCtx.fillRect(i * bw + 1, H - h, bw - 2, h);
  }

  // Freq labels
  fCtx.fillStyle = COLORS.text2;
  fCtx.font = '9px Share Tech Mono, monospace';
  const freqLabels = ['0', '1k', '2k', '4k', '8k', '12k', '16k', '20k'];
  freqLabels.forEach((l, i) => {
    fCtx.fillText(l, (i / (freqLabels.length - 1)) * (W - 20), H - 2);
  });
}

// ── Generate normal signal ──
function generateNormalWave() {
  const phase = frameCount * 0.04;
  for (let i = 0; i < waveData.length; i++) {
    const t = i / waveData.length;
    waveData[i] = (
      Math.sin(2 * Math.PI * t * 3 + phase) * 0.08 +
      Math.sin(2 * Math.PI * t * 7 + phase * 1.3) * 0.04 +
      (Math.random() - 0.5) * 0.06
    );
  }
  for (let i = 0; i < fftData.length; i++) {
    const target = i < 10 ? 0.15 + Math.random() * 0.1
                 : i < 20 ? 0.08 + Math.random() * 0.06
                 :           0.02 + Math.random() * 0.03;
    fftData[i] += (target - fftData[i]) * 0.2;
  }
}

// ── Generate anomaly signal ──
function generateAnomalyWave() {
  const burst = Math.sin(frameCount * 0.3) * 0.5 + 0.5;
  for (let i = 0; i < waveData.length; i++) {
    const t = i / waveData.length;
    waveData[i] = (
      Math.sin(2 * Math.PI * t * 12 + frameCount * 0.2) * 0.35 * burst +
      Math.sin(2 * Math.PI * t * 5  + frameCount * 0.5) * 0.25 +
      (Math.random() - 0.5) * 0.3 * burst
    );
  }
  for (let i = 0; i < fftData.length; i++) {
    const anomalyBand = i > 20 && i < 45;
    const target = anomalyBand
      ? 0.5 + Math.random() * 0.4
      : 0.1 + Math.random() * 0.15;
    fftData[i] += (target - fftData[i]) * 0.3;
  }
}

// ── Update stat cards ──
function updateStats() {
  const maxAmp = Math.max(...Array.from(waveData).map(Math.abs));
  const db  = Math.round(20 * Math.log10(maxAmp + 0.001));
  const rms = Math.sqrt(waveData.reduce((a, v) => a + v * v, 0) / waveData.length);
  const maxIdx = fftData.indexOf(Math.max(...fftData));
  const freq = Math.round((maxIdx / fftData.length) * 20000);

  document.getElementById('statAmplitude').innerHTML = db + '<span class="stat-unit">dB</span>';
  document.getElementById('statFreq').innerHTML      = freq + '<span class="stat-unit">Hz</span>';
  document.getElementById('statRMS').innerHTML       = rms.toFixed(3) + '<span class="stat-unit">V</span>';
}

// ── Trigger anomaly (called from HTML or externally) ──
function triggerAnomaly() {
  if (!simActive) {
    addLog('⚠ Khởi động mô phỏng trước khi kích hoạt bất thường', 'warn');
    return;
  }
  anomalyMode    = true;
  anomalyTimer   = 120;
  anomalyTypeIdx = Math.random() > 0.5 ? 3 : 4;

  const label = SOUND_LABELS[anomalyTypeIdx];
  const level = anomalyTypeIdx === 4 ? 'danger' : 'warning';

  setAlert(level,
    anomalyTypeIdx === 4 ? 'NGUY HIỂM' : 'CẢNH BÁO',
    `Phát hiện: ${label}`
  );
  addLog(`[AI] Phát hiện âm thanh bất thường: "${label}"`, 'err');

  if (relayStates[4]) {
    setTimeout(() => {
      relayStates[1] = false;
      document.getElementById('relay1').classList.replace('on', 'off');
      addLog('[AI] Auto-cut: Relay 1 → TẮT (bảo vệ thiết bị)', 'warn');
    }, 600);
  }
}

// ── Main render loop ──
function gameLoop() {
  if (!simActive) { requestAnimationFrame(gameLoop); return; }

  frameCount++;

  // if (anomalyTimer > 0) {
  //   anomalyTimer--;
  //   // generateAnomalyWave();

  //   const confs = [0.04, 0.03, 0.05, 0, 0];
  //   confs[anomalyTypeIdx] = 0.75 + Math.random() * 0.2;
  //   const rest = 1 - confs[anomalyTypeIdx];
  //   [0, 1, 2]
  //     .filter(i => i !== anomalyTypeIdx && i < 3)
  //     .forEach((i, j) => { confs[i] = rest * [0.5, 0.3, 0.2][j]; });
  //   updateSoundTags(confs, anomalyTypeIdx);

  // } else {
  //   if (anomalyMode) {
  //     anomalyMode = false;
  //     setAlert('normal', 'BÌNH THƯỜNG', 'Không phát hiện âm thanh bất thường');
  //     addLog('[AI] Trạng thái: Trở về bình thường', 'ok');
  //     relayStates[1] = true;
  //     document.getElementById('relay1').classList.replace('off', 'on');
  //     addLog('[AI] Auto-restore: Relay 1 → BẬT', 'ok');
  //     updateSoundTags([0.85, 0.05, 0.05, 0.03, 0.02], -1);
  //   }
  //   // generateNormalWave();

  //   if (frameCount % 80 === 0) {
  //     if (Math.random() < 0.3) {
  //       updateSoundTags([0.60, 0.25, 0.10, 0.03, 0.02], 1);
  //     } else {
  //       updateSoundTags([0.85, 0.05, 0.05, 0.03, 0.02], 0);
  //     }
  //   }
  // }

  if (frameCount % 3 === 0) updateStats();
  drawWaveform();
  drawFFT();

  requestAnimationFrame(gameLoop);
}

// ── Start simulation ──
function startSimulation() {
  if (simActive) return;
  simActive = true;

  document.getElementById('connDot').style.background  = COLORS.green;
  document.getElementById('connDot').style.boxShadow   = '0 0 6px ' + COLORS.green;
  document.getElementById('connLabel').textContent      = 'ĐÃ KẾT NỐI';

  addLog('[SYS] Kết nối ESP32 thành công — SSID: ESP32_AUDIO_01', 'ok');
  addLog('[SYS] Mẫu âm thanh: 44100Hz / 16-bit / Mono', 'ok');
  addLog('[AI]  Mô hình: SoundNet-Lite v2.1 (nạp vào SPIFFS)', 'ok');
  addLog('[SYS] Bắt đầu thu âm thanh...', 'ok');

  updateSoundTags([0.85, 0.05, 0.05, 0.03, 0.02], 0);
  setAlert('normal', 'BÌNH THƯỜNG', 'Hệ thống đang giám sát');
  requestAnimationFrame(gameLoop);
}

// ── Init ──
resizeCanvases();
addLog('[SYS] Khởi động hệ thống...', 'ok');
addLog('[SYS] Nhấn ▶ MÔ PHỎNG để bắt đầu', '');
document.getElementById('connLabel').textContent    = 'CHỜ KẾT NỐI';
document.getElementById('connDot').style.background = COLORS.yellow;
document.getElementById('connDot').style.boxShadow  = '0 0 6px ' + COLORS.yellow;
requestAnimationFrame(gameLoop);

// =============================
// REAL-TIME FROM PYTHON (ADD)
// =============================
const socket = io('http://192.168.1.17:5000');

socket.on('connect', () => {
  console.log("Connected to AI server");

  document.getElementById('connDot').style.background = '#00e57d';
  document.getElementById('connDot').style.boxShadow  = '0 0 6px #00e57d';
  document.getElementById('connLabel').textContent = 'AI CONNECTED';

  addLog('[SYS] Kết nối AI server thành công', 'ok');
});

socket.on('audio_event', (data) => {
  const { label, class_id, confidence, confs } = data;

  // cập nhật thanh %
  updateSoundTags(confs, class_id);

  // cảnh báo
  if (confidence > 0.5 && class_id !== 0) {
    const level = (class_id === 3) ? 'danger' : 'warning';

    setAlert(level, 'CẢNH BÁO', `Phát hiện: ${label}`);
    addLog(`[AI] ${label} (${(confidence * 100).toFixed(1)}%)`, 'err');
  } else {
    setAlert('normal', 'BÌNH THƯỜNG', 'Không phát hiện bất thường');
  }
});

function startSystem() {
  addLog('[SYS] Bắt đầu hệ thống AI...', 'ok');
}

