const video = document.getElementById('video');
const canvas = document.getElementById('gl');
const overlay = document.getElementById('overlay');
const statusEl = document.getElementById('status');
const pointsEl = document.getElementById('points');
const sharpnessEl = document.getElementById('sharpness');

const params = {
  brightMin: 0.25,
  brightMax: 0.75,
  blueMin: 0.05,
  blueMax: 0.25,
  satMin: 0.05,
  satMax: 0.35,
  highpassMix: 0.0,
  highpassGain: 1.0,
  dogMix: 0.0,
  dogGain: 1.0,
};

let gl;
let program;
let tex;
let vbo;
let uniforms = {};
let overlayCtx;
let nmsCanvas;
let nmsCtx;
let nmsWidth = 320;
let nmsHeight = 180;
let lastConstraintsApplied = false;
let currentStream = null;
let currentResolution = { width: 1920, height: 1080 };
let isRestarting = false;

const nmsConfig = {
  threshold: 0.4,
  radius: 3,
  maxPoints: 20,
};

const vertSrc = `
attribute vec2 aPos;
varying vec2 vUv;
void main() {
  // Flip Y only to correct upside-down without mirroring left-right.
  vUv = vec2(aPos.x, -aPos.y) * 0.5 + 0.5;
  gl_Position = vec4(aPos, 0.0, 1.0);
}
`;

const fragSrc = `
precision mediump float;

uniform sampler2D uTex;
uniform vec2 uTexSize;
uniform float uBrightMin;
uniform float uBrightMax;
uniform float uBlueMin;
uniform float uBlueMax;
uniform float uSatMin;
uniform float uSatMax;
uniform float uHighpassMix;
uniform float uHighpassGain;

varying vec2 vUv;

void main() {
  vec3 rgb = texture2D(uTex, vUv).rgb;
  float brightness = max(max(rgb.r, rgb.g), rgb.b);
  float blueDiff = rgb.b - (rgb.r + rgb.g) * 0.5;
  float saturation = max(max(rgb.r, rgb.g), rgb.b) - min(min(rgb.r, rgb.g), rgb.b);

  vec2 texel = 1.0 / uTexSize;
  vec3 c00 = texture2D(uTex, vUv + texel * vec2(-1.0, -1.0)).rgb;
  vec3 c01 = texture2D(uTex, vUv + texel * vec2( 0.0, -1.0)).rgb;
  vec3 c02 = texture2D(uTex, vUv + texel * vec2( 1.0, -1.0)).rgb;
  vec3 c10 = texture2D(uTex, vUv + texel * vec2(-1.0,  0.0)).rgb;
  vec3 c12 = texture2D(uTex, vUv + texel * vec2( 1.0,  0.0)).rgb;
  vec3 c20 = texture2D(uTex, vUv + texel * vec2(-1.0,  1.0)).rgb;
  vec3 c21 = texture2D(uTex, vUv + texel * vec2( 0.0,  1.0)).rgb;
  vec3 c22 = texture2D(uTex, vUv + texel * vec2( 1.0,  1.0)).rgb;

  float b00 = max(max(c00.r, c00.g), c00.b);
  float b01 = max(max(c01.r, c01.g), c01.b);
  float b02 = max(max(c02.r, c02.g), c02.b);
  float b10 = max(max(c10.r, c10.g), c10.b);
  float b11 = brightness;
  float b12 = max(max(c12.r, c12.g), c12.b);
  float b20 = max(max(c20.r, c20.g), c20.b);
  float b21 = max(max(c21.r, c21.g), c21.b);
  float b22 = max(max(c22.r, c22.g), c22.b);

  float localAvg = (b00 + b01 + b02 + b10 + b11 + b12 + b20 + b21 + b22) / 9.0;
  float highpass = max(brightness - localAvg, 0.0) * uHighpassGain;

  float brightnessGate = smoothstep(uBrightMin, uBrightMax, mix(brightness, highpass, uHighpassMix));
  float blueGate = smoothstep(uBlueMin, uBlueMax, blueDiff);
  float satGate = smoothstep(uSatMin, uSatMax, saturation);

  float mask = brightnessGate * blueGate * satGate;

  // Show raw camera feed to avoid colored mask overlay.
  gl_FragColor = vec4(rgb, 1.0);
}
`;

function createShader(gl, type, src) {
  const shader = gl.createShader(type);
  gl.shaderSource(shader, src);
  gl.compileShader(shader);
  if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
    throw new Error(gl.getShaderInfoLog(shader));
  }
  return shader;
}

function createProgram(gl, vs, fs) {
  const program = gl.createProgram();
  gl.attachShader(program, vs);
  gl.attachShader(program, fs);
  gl.linkProgram(program);
  if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
    throw new Error(gl.getProgramInfoLog(program));
  }
  return program;
}

function initGL() {
  gl = canvas.getContext('webgl', {
    alpha: false,
    antialias: false,
    depth: false,
    stencil: false,
    preserveDrawingBuffer: false,
  });
  if (!gl) throw new Error('WebGL not supported');

  const vs = createShader(gl, gl.VERTEX_SHADER, vertSrc);
  const fs = createShader(gl, gl.FRAGMENT_SHADER, fragSrc);
  program = createProgram(gl, vs, fs);
  gl.useProgram(program);

  const quad = new Float32Array([
    -1, -1,
     1, -1,
    -1,  1,
    -1,  1,
     1, -1,
     1,  1,
  ]);

  vbo = gl.createBuffer();
  gl.bindBuffer(gl.ARRAY_BUFFER, vbo);
  gl.bufferData(gl.ARRAY_BUFFER, quad, gl.STATIC_DRAW);

  const aPos = gl.getAttribLocation(program, 'aPos');
  gl.enableVertexAttribArray(aPos);
  gl.vertexAttribPointer(aPos, 2, gl.FLOAT, false, 0, 0);

  tex = gl.createTexture();
  gl.bindTexture(gl.TEXTURE_2D, tex);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);

  uniforms.uTex = gl.getUniformLocation(program, 'uTex');
  uniforms.uTexSize = gl.getUniformLocation(program, 'uTexSize');
  uniforms.uBrightMin = gl.getUniformLocation(program, 'uBrightMin');
  uniforms.uBrightMax = gl.getUniformLocation(program, 'uBrightMax');
  uniforms.uBlueMin = gl.getUniformLocation(program, 'uBlueMin');
  uniforms.uBlueMax = gl.getUniformLocation(program, 'uBlueMax');
  uniforms.uSatMin = gl.getUniformLocation(program, 'uSatMin');
  uniforms.uSatMax = gl.getUniformLocation(program, 'uSatMax');
  uniforms.uHighpassMix = gl.getUniformLocation(program, 'uHighpassMix');
  uniforms.uHighpassGain = gl.getUniformLocation(program, 'uHighpassGain');

  gl.uniform1i(uniforms.uTex, 0);
}

function updateUniforms() {
  gl.uniform1f(uniforms.uBrightMin, params.brightMin);
  gl.uniform1f(uniforms.uBrightMax, params.brightMax);
  gl.uniform1f(uniforms.uBlueMin, params.blueMin);
  gl.uniform1f(uniforms.uBlueMax, params.blueMax);
  gl.uniform1f(uniforms.uSatMin, params.satMin);
  gl.uniform1f(uniforms.uSatMax, params.satMax);
  gl.uniform1f(uniforms.uHighpassMix, params.highpassMix);
  gl.uniform1f(uniforms.uHighpassGain, params.highpassGain);
}

function resize() {
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  const w = Math.floor(window.innerWidth * dpr);
  const h = Math.floor(window.innerHeight * dpr);
  if (canvas.width !== w || canvas.height !== h) {
    canvas.width = w;
    canvas.height = h;
    gl.viewport(0, 0, w, h);
  }
  if (overlay.width !== w || overlay.height !== h) {
    overlay.width = w;
    overlay.height = h;
  }
}

function setupNms() {
  overlayCtx = overlay.getContext('2d');
  nmsCanvas = document.createElement('canvas');
  nmsCtx = nmsCanvas.getContext('2d', { willReadFrequently: true });
  nmsCanvas.width = nmsWidth;
  nmsCanvas.height = nmsHeight;
}

function computeNmsPoints() {
  if (!video.videoWidth || !video.videoHeight) return [];

  const aspect = video.videoWidth / video.videoHeight;
  nmsWidth = 360;
  nmsHeight = Math.round(nmsWidth / aspect);
  if (nmsCanvas.width !== nmsWidth || nmsCanvas.height !== nmsHeight) {
    nmsCanvas.width = nmsWidth;
    nmsCanvas.height = nmsHeight;
  }

  nmsCtx.drawImage(video, 0, 0, nmsWidth, nmsHeight);
  const img = nmsCtx.getImageData(0, 0, nmsWidth, nmsHeight);
  const data = img.data;
  const total = nmsWidth * nmsHeight;

  const brightness = new Float32Array(total);
  const blueDiff = new Float32Array(total);
  const saturation = new Float32Array(total);
  const mask = new Float32Array(total);

  for (let i = 0, p = 0; i < total; i++, p += 4) {
    const r = data[p] / 255;
    const g = data[p + 1] / 255;
    const b = data[p + 2] / 255;
    const maxc = Math.max(r, g, b);
    const minc = Math.min(r, g, b);
    brightness[i] = maxc;
    blueDiff[i] = b - (r + g) * 0.5;
    saturation[i] = maxc - minc;
  }

  const w = nmsWidth;
  const h = nmsHeight;
  const blurSmall = new Float32Array(total);
  const blurLarge = new Float32Array(total);
  if (params.dogMix > 0) {
    boxBlur(brightness, blurSmall, w, h, 1);
    boxBlur(brightness, blurLarge, w, h, 3);
  }
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const idx = y * w + x;
      let localAvg = brightness[idx];
      if (params.highpassMix > 0 && x > 0 && y > 0 && x < w - 1 && y < h - 1) {
        let sum = 0;
        for (let oy = -1; oy <= 1; oy++) {
          for (let ox = -1; ox <= 1; ox++) {
            sum += brightness[(y + oy) * w + (x + ox)];
          }
        }
        localAvg = sum / 9;
      }
      const highpass = Math.max(brightness[idx] - localAvg, 0) * params.highpassGain;
      const dog = params.dogMix > 0
        ? Math.max(blurSmall[idx] - blurLarge[idx], 0) * params.dogGain
        : 0;

      let brightMix = brightness[idx];
      if (params.highpassMix > 0) {
        brightMix = brightMix * (1 - params.highpassMix) + highpass * params.highpassMix;
      }
      if (params.dogMix > 0) {
        brightMix = brightMix * (1 - params.dogMix) + dog * params.dogMix;
      }

      const brightnessGate = smoothstep(params.brightMin, params.brightMax, brightMix);
      const blueGate = smoothstep(params.blueMin, params.blueMax, blueDiff[idx]);
      const satGate = smoothstep(params.satMin, params.satMax, saturation[idx]);
      mask[idx] = brightnessGate * blueGate * satGate;
    }
  }

  const points = [];
  const r = nmsConfig.radius;
  for (let y = r; y < h - r; y++) {
    for (let x = r; x < w - r; x++) {
      const idx = y * w + x;
      const v = mask[idx];
      if (v < nmsConfig.threshold) continue;
      let isMax = true;
      for (let oy = -r; oy <= r && isMax; oy++) {
        for (let ox = -r; ox <= r; ox++) {
          if (ox === 0 && oy === 0) continue;
          if (mask[(y + oy) * w + (x + ox)] > v) {
            isMax = false;
            break;
          }
        }
      }
      if (!isMax) continue;
      let sum = 0;
      let sx = 0;
      let sy = 0;
      for (let oy = -1; oy <= 1; oy++) {
        for (let ox = -1; ox <= 1; ox++) {
          const val = mask[(y + oy) * w + (x + ox)];
          sum += val;
          sx += (x + ox) * val;
          sy += (y + oy) * val;
        }
      }
      const px = sum > 0 ? sx / sum : x;
      const py = sum > 0 ? sy / sum : y;
      points.push({ x: px, y: py, score: v });
      if (points.length >= nmsConfig.maxPoints) break;
    }
    if (points.length >= nmsConfig.maxPoints) break;
  }

  return { points, sharpness: computeSharpness(brightness, w, h) };
}

function drawOverlay(points) {
  if (!overlayCtx) return;
  overlayCtx.clearRect(0, 0, overlay.width, overlay.height);

  const sx = overlay.width / nmsWidth;
  const sy = overlay.height / nmsHeight;
  overlayCtx.lineWidth = 2;
  overlayCtx.strokeStyle = 'rgba(255, 255, 255, 0.9)';
  overlayCtx.fillStyle = 'rgba(74, 163, 255, 0.85)';

  for (const pt of points) {
    const x = pt.x * sx;
    const y = pt.y * sy;
    overlayCtx.beginPath();
    overlayCtx.arc(x, y, 5, 0, Math.PI * 2);
    overlayCtx.stroke();
    overlayCtx.beginPath();
    overlayCtx.arc(x, y, 2.5, 0, Math.PI * 2);
    overlayCtx.fill();
  }
  pointsEl.textContent = `Points: ${points.length}`;
}

function render() {
  if (video.readyState >= 2) {
    gl.bindTexture(gl.TEXTURE_2D, tex);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGB, gl.RGB, gl.UNSIGNED_BYTE, video);
    if (video.videoWidth && video.videoHeight) {
      gl.uniform2f(uniforms.uTexSize, video.videoWidth, video.videoHeight);
    }
    updateUniforms();
    gl.drawArrays(gl.TRIANGLES, 0, 6);
    const result = computeNmsPoints();
    drawOverlay(result.points);
    sharpnessEl.textContent = `Sharpness: ${result.sharpness.toFixed(3)}`;
  }
  requestAnimationFrame(render);
}

async function startCamera() {
  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    const hint = location.protocol === 'https:'
      ? '瀏覽器未支援 getUserMedia'
      : '需要 HTTPS 才能使用相機（手機不支援 http）';
    throw new Error(hint);
  }

  const constraints = {
    audio: false,
    video: {
      facingMode: { ideal: 'environment' },
      width: { ideal: currentResolution.width },
      height: { ideal: currentResolution.height },
    },
  };

  if (currentStream) {
    currentStream.getTracks().forEach((t) => t.stop());
  }

  const stream = await navigator.mediaDevices.getUserMedia(constraints);
  currentStream = stream;
  video.srcObject = stream;
  applyVideoConstraints(stream);
  await video.play();
}

async function main() {
  try {
    initGL();
    setupNms();
    resize();
    window.addEventListener('resize', resize);
    initControls();
    await startCamera();
    statusEl.textContent = 'Running';
    render();
  } catch (err) {
    statusEl.textContent = `Error: ${err.message}`;
  }
}

main();

function initControls() {
  const bindings = [
    ['brightMin', 'brightMinVal'],
    ['brightMax', 'brightMaxVal'],
    ['blueMin', 'blueMinVal'],
    ['blueMax', 'blueMaxVal'],
    ['satMin', 'satMinVal'],
    ['satMax', 'satMaxVal'],
    ['highpassMix', 'highpassMixVal'],
    ['highpassGain', 'highpassGainVal'],
    ['dogMix', 'dogMixVal'],
    ['dogGain', 'dogGainVal'],
  ];

  bindings.forEach(([id, valId]) => {
    const input = document.getElementById(id);
    const label = document.getElementById(valId);
    if (!input || !label) return;

    const apply = () => {
      const value = Number.parseFloat(input.value);
      params[id] = Number.isFinite(value) ? value : params[id];
      label.textContent = params[id].toFixed(2);
    };

    input.addEventListener('input', apply);
    apply();
  });

  const resolutionSelect = document.getElementById('resolution');
  const resolutionVal = document.getElementById('resolutionVal');
  if (resolutionSelect && resolutionVal) {
    const applyResolution = async () => {
      const [w, h] = resolutionSelect.value.split('x').map((v) => Number.parseInt(v, 10));
      if (Number.isFinite(w) && Number.isFinite(h)) {
        currentResolution = { width: w, height: h };
        resolutionVal.textContent = `${w}x${h}`;
        lastConstraintsApplied = false;
        await startCamera();
      }
    };
    resolutionSelect.addEventListener('change', () => {
      applyResolution().catch(() => {});
    });
    applyResolution().catch(() => {});
  }

  const refocusBtn = document.getElementById('refocusBtn');
  const refocusVal = document.getElementById('refocusVal');
  if (refocusBtn && refocusVal) {
    refocusBtn.addEventListener('click', () => {
      if (isRestarting) return;
      isRestarting = true;
      refocusVal.textContent = 'restarting…';
      lastConstraintsApplied = false;
      startCamera()
        .then(() => {
          refocusVal.textContent = 'ready';
        })
        .catch(() => {
          refocusVal.textContent = 'failed';
        })
        .finally(() => {
          isRestarting = false;
        });
    });
  }
}

function smoothstep(edge0, edge1, x) {
  const t = Math.max(0, Math.min(1, (x - edge0) / (edge1 - edge0)));
  return t * t * (3 - 2 * t);
}

function boxBlur(src, dst, w, h, radius) {
  const size = radius * 2 + 1;
  const tmp = new Float32Array(w * h);

  for (let y = 0; y < h; y++) {
    let sum = 0;
    for (let x = -radius; x <= radius; x++) {
      const xx = Math.min(w - 1, Math.max(0, x));
      sum += src[y * w + xx];
    }
    for (let x = 0; x < w; x++) {
      tmp[y * w + x] = sum / size;
      const x0 = x - radius;
      const x1 = x + radius + 1;
      const v0 = src[y * w + Math.min(w - 1, Math.max(0, x0))];
      const v1 = src[y * w + Math.min(w - 1, Math.max(0, x1))];
      sum += v1 - v0;
    }
  }

  for (let x = 0; x < w; x++) {
    let sum = 0;
    for (let y = -radius; y <= radius; y++) {
      const yy = Math.min(h - 1, Math.max(0, y));
      sum += tmp[yy * w + x];
    }
    for (let y = 0; y < h; y++) {
      dst[y * w + x] = sum / size;
      const y0 = y - radius;
      const y1 = y + radius + 1;
      const v0 = tmp[Math.min(h - 1, Math.max(0, y0)) * w + x];
      const v1 = tmp[Math.min(h - 1, Math.max(0, y1)) * w + x];
      sum += v1 - v0;
    }
  }
}

function computeSharpness(src, w, h) {
  let sum = 0;
  let count = 0;
  for (let y = 1; y < h - 1; y++) {
    for (let x = 1; x < w - 1; x++) {
      const idx = y * w + x;
      const lap =
        4 * src[idx] -
        src[idx - 1] -
        src[idx + 1] -
        src[idx - w] -
        src[idx + w];
      sum += lap * lap;
      count++;
    }
  }
  return count > 0 ? sum / count : 0;
}

function applyVideoConstraints(stream) {
  if (lastConstraintsApplied) return;
  const track = stream.getVideoTracks()[0];
  if (!track) return;
  const caps = track.getCapabilities ? track.getCapabilities() : {};
  const constraints = {};

  if (caps.focusMode && caps.focusMode.includes('continuous')) {
    constraints.focusMode = 'continuous';
  }
  if (caps.exposureMode && caps.exposureMode.includes('continuous')) {
    constraints.exposureMode = 'continuous';
  }
  if (caps.whiteBalanceMode && caps.whiteBalanceMode.includes('continuous')) {
    constraints.whiteBalanceMode = 'continuous';
  }
  if (caps.torch) {
    constraints.torch = false;
  }

  if (Object.keys(constraints).length > 0) {
    track.applyConstraints({ advanced: [constraints] }).catch(() => {});
    lastConstraintsApplied = true;
  }
}
