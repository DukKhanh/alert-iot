// ── Canvas Setup ──
const waveCanvas = document.getElementById('waveCanvas');
const fftCanvas  = document.getElementById('fftCanvas');
const specCanvas = document.getElementById('specCanvas');
const wCtx = waveCanvas.getContext('2d');
const fCtx  = fftCanvas.getContext('2d');
const sCtx  = specCanvas.getContext('2d');

const COLORS = {
  bg: '#0a0c10', bg2: '#0d1014', bg3: '#11141a',
  grid: '#1a1f2a', green: '#00e57d', red: '#ff3b3b',
  yellow: '#ffd94a', cyan: '#00d4ff', text2: '#6b7a94', blue: '#3d9bff'
};

const SOUND_LABELS = ['Background', 'Glass Breaking', 'Gunshot', 'Scream'];
const ANOMALY_LABELS = ['Glass Breaking', 'Gunshot', 'Scream'];

// ── State ──
let waveData = new Float32Array(1024).fill(0);
let fftData  = new Float32Array(64).fill(0);
let spectrogramData = null;   // { data: [[...]], shape: [n_mels, t], db_min, db_max }
let serverWaveReady = false;

// BỔ SUNG THÊM 3 DÒNG NÀY:
let simActive = false;
let anomalyMode = false;
let frameCount = 0;


// ── Canvas resize ──
function resizeCanvases() {
  waveCanvas.width  = waveCanvas.offsetWidth;
  waveCanvas.height = waveCanvas.offsetHeight;
  fftCanvas.width   = fftCanvas.offsetWidth;
  fftCanvas.height  = fftCanvas.offsetHeight;
  specCanvas.width  = specCanvas.offsetWidth;
  specCanvas.height = specCanvas.offsetHeight;
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

  wCtx.strokeStyle = COLORS.grid;
  wCtx.lineWidth = 0.5;
  for (let x = 0; x < W; x += W / 8) {
    wCtx.beginPath(); wCtx.moveTo(x, 0); wCtx.lineTo(x, H); wCtx.stroke();
  }
  for (let y = 0; y < H; y += H / 4) {
    wCtx.beginPath(); wCtx.moveTo(0, y); wCtx.lineTo(W, y); wCtx.stroke();
  }
  wCtx.strokeStyle = '#1f2635';
  wCtx.lineWidth = 1;
  wCtx.beginPath(); wCtx.moveTo(0, H / 2); wCtx.lineTo(W, H / 2); wCtx.stroke();

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

  fCtx.strokeStyle = COLORS.grid;
  fCtx.lineWidth = 0.5;
  for (let y = 0; y < H; y += H / 4) {
    fCtx.beginPath(); fCtx.moveTo(0, y); fCtx.lineTo(W, y); fCtx.stroke();
  }

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

  fCtx.fillStyle = COLORS.text2;
  fCtx.font = '9px Share Tech Mono, monospace';
  const freqLabels = ['0', '1k', '2k', '4k', '8k', '12k', '16k', '20k'];
  freqLabels.forEach((l, i) => {
    fCtx.fillText(l, (i / (freqLabels.length - 1)) * (W - 20), H - 2);
  });
}

// ── Draw Mel-Spectrogram ──
// data: mảng 2D [n_mels][time], giá trị [0,1] đã normalize
function drawSpectrogram() {
  const W = specCanvas.width, H = specCanvas.height;
  if (!spectrogramData || !spectrogramData.data.length) {
    sCtx.fillStyle = COLORS.bg;
    sCtx.fillRect(0, 0, W, H);
    sCtx.fillStyle = COLORS.text2;
    sCtx.font = '11px Share Tech Mono, monospace';
    sCtx.fillText('Chờ dữ liệu spectrogram...', W / 2 - 90, H / 2);
    return;
  }

  const melBands = spectrogramData.shape[0];   // 128
  const timeCols = spectrogramData.shape[1];
  const cellW = W / timeCols;
  const cellH = H / melBands;

  for (let t = 0; t < timeCols; t++) {
    for (let m = 0; m < melBands; m++) {
      const val = spectrogramData.data[m][t];  // [0,1]
      // Viridis-like colormap: thấp=tím, cao=vàng
      const r = Math.round(Math.min(255, val * 2.5 * 255));
      const g = Math.round(Math.min(255, Math.max(0, (val - 0.3) * 2 * 255)));
      const b = Math.round(Math.max(0, (1 - val * 1.5) * 255));
      sCtx.fillStyle = `rgb(${r},${g},${b})`;
      // mel band 0 = thấp nhất → vẽ từ dưới lên
      sCtx.fillRect(
        Math.floor(t * cellW),
        Math.floor((melBands - 1 - m) * cellH),
        Math.ceil(cellW) + 1,
        Math.ceil(cellH) + 1
      );
    }
  }

  // Trục tần số (mel bands)
  sCtx.fillStyle = 'rgba(0,0,0,0.5)';
  sCtx.fillRect(0, 0, 36, H);
  sCtx.fillStyle = COLORS.text2;
  sCtx.font = '9px Share Tech Mono, monospace';
  const freqTicks = ['8k', '4k', '2k', '1k', '500', '250', '0'];
  freqTicks.forEach((l, i) => {
    const y = (i / (freqTicks.length - 1)) * H;
    sCtx.fillText(l, 2, y + 4);
  });
}

// ── Tính FFT đơn giản từ waveform thật ──
// Server gửi samples thô (float), ta tính năng lượng theo dải tần (band energy)
function computeFFTFromWave() {
  const N = waveData.length;
  for (let b = 0; b < fftData.length; b++) {
    const start = Math.floor(b * N / fftData.length);
    const end   = Math.floor((b + 1) * N / fftData.length);
    let energy = 0;
    for (let i = start; i < end; i++) {
      energy += waveData[i] * waveData[i];
    }
    const rms = Math.sqrt(energy / Math.max(end - start, 1));
    // Smooth để tránh nhấp nháy
    fftData[b] += (Math.min(rms * 5, 1.0) - fftData[b]) * 0.3;
  }
}

// ── Update stat cards ──
function updateStats() {
  const maxAmp = Math.max(...Array.from(waveData).map(Math.abs));
  const db  = Math.round(20 * Math.log10(maxAmp + 0.001));
  const rms = Math.sqrt(waveData.reduce((a, v) => a + v * v, 0) / waveData.length);
  const maxIdx = Array.from(fftData).indexOf(Math.max(...fftData));
  const freq = Math.round((maxIdx / fftData.length) * 20000);

  document.getElementById('statAmplitude').innerHTML = db + '<span class="stat-unit">dB</span>';
  document.getElementById('statFreq').innerHTML      = freq + '<span class="stat-unit">Hz</span>';
  document.getElementById('statRMS').innerHTML       = rms.toFixed(3) + '<span class="stat-unit">V</span>';
}

// ── Main render loop ──
// Không còn generate random — chỉ draw những gì server gửi
function gameLoop() {
  if (!simActive) { requestAnimationFrame(gameLoop); return; }

  frameCount++;

  // Chờ frame đầu tiên từ server trước khi vẽ
  if (!serverWaveReady) {
    requestAnimationFrame(gameLoop);
    return;
  }

  if (frameCount % 3 === 0) updateStats();
  drawWaveform();
  drawFFT();
  drawSpectrogram();

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
  addLog('[SYS] Mẫu âm thanh: 16000Hz / 32-bit / Mono', 'ok');
  addLog('[AI]  Mô hình: spectrogram_model_best.keras', 'ok');
  addLog('[SYS] Bắt đầu thu âm thanh...', 'ok');

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
// REAL-TIME TỪ PYTHON SERVER
// =============================
const socket = io('http://192.168.1.35:5000');

socket.on('connect', () => {
  console.log("Connected to AI server");

  document.getElementById('connDot').style.background = '#00e57d';
  document.getElementById('connDot').style.boxShadow  = '0 0 6px #00e57d';
  document.getElementById('connLabel').textContent = 'AI CONNECTED';

  addLog('[SYS] Kết nối AI server thành công', 'ok');
});

socket.on('disconnect', () => {
  document.getElementById('connDot').style.background = COLORS.yellow;
  document.getElementById('connDot').style.boxShadow  = '0 0 6px ' + COLORS.yellow;
  document.getElementById('connLabel').textContent = 'MẤT KẾT NỐI';
  addLog('[SYS] Mất kết nối AI server', 'warn');
  serverWaveReady = false;
});

// Nhận waveform thật từ server (thay thế generateNormalWave / generateAnomalyWave)
socket.on('waveform', (data) => {
  waveData = new Float32Array(data.samples);
  computeFFTFromWave();
  serverWaveReady = true;
  if (!simActive) startSimulation();
});

// Nhận mel-spectrogram realtime từ server (emit mỗi 200ms)
socket.on('spectrogram', (data) => {
  spectrogramData = data;
  document.getElementById('specDbMin').textContent = data.db_min + ' dB';
  document.getElementById('specDbMax').textContent = data.db_max + ' dB';
});

// Nhận kết quả phân loại — server quyết định anomalyMode và UI
socket.on('audio_event', (data) => {
  const { label, class_id, confidence, confs } = data;

  updateSoundTags(confs, class_id);

  if (confidence > 0.5 && class_id !== 0) {
    const level = (class_id === 3) ? 'danger' : 'warning';
    anomalyMode = true;
    setAlert(level, 'CẢNH BÁO', `Phát hiện: ${label}`);
    addLog(`[AI] ${label} (${(confidence * 100).toFixed(1)}%)`, 'err');
  } else {
    anomalyMode = false;
    setAlert('normal', 'BÌNH THƯỜNG', 'Không phát hiện bất thường');
  }
});

function startSystem() {
  addLog('[SYS] Bắt đầu hệ thống AI...', 'ok');
}