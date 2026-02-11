const video = document.getElementById('video');
const canvas = document.getElementById('gl');
const overlay = document.getElementById('overlay');
const statusEl = document.getElementById('status');
const pointsEl = document.getElementById('points');
const sharpnessEl = document.getElementById('sharpness');
const lockEl = document.getElementById('lock');
const scoreEl = document.getElementById('score');
const poseEl = document.getElementById('pose');

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
let focusCaps = { min: 0, max: 1 };
let lastInfoTime = 0;
let exposureCaps = { min: 1, max: 33 };
let isoCaps = { min: 50, max: 800 };
let expCompCaps = { min: -2, max: 2 };
let lastLock = false;
let lastBeepTime = 0;
let filteredPose = null;
let viewRect = { x: 0, y: 0, w: 0, h: 0 };
let prevPoints = [];
let cropRect = { sx: 0, sy: 0, sw: 0, sh: 0 };

const kalmanConfig = {
  alpha: 0.35,
};

const cvReady = new Promise((resolve) => {
  if (typeof cv !== 'undefined' && cv.Mat) {
    resolve();
    return;
  }
  const check = () => {
    if (typeof cv !== 'undefined' && cv.Mat) {
      resolve();
    } else {
      setTimeout(check, 50);
    }
  };
  check();
});

const nmsConfig = {
  threshold: 0.28,
  radius: 3,
  maxPoints: 32,
};

const objectPoints = [
  [62, 43.3, 0],    // LED1
  [62, -43.3, 0],   // LED2
  [-62, -43.3, 0],  // LED3
  [-62, 43.3, 0],   // LED4
  [0, 151.6, 40],   // LED5
];

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
uniform vec4 uSrcRect;
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
  vec2 uv = uSrcRect.xy + vUv * uSrcRect.zw;
  vec3 rgb = texture2D(uTex, uv).rgb;
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
  uniforms.uSrcRect = gl.getUniformLocation(program, 'uSrcRect');
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
  gl.uniform4f(
    uniforms.uSrcRect,
    cropRect.sx / video.videoWidth || 0,
    cropRect.sy / video.videoHeight || 0,
    cropRect.sw / video.videoWidth || 1,
    cropRect.sh / video.videoHeight || 1
  );
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

  const aspect = cropRect.sw && cropRect.sh
    ? cropRect.sw / cropRect.sh
    : video.videoWidth / video.videoHeight;
  nmsWidth = 360;
  nmsHeight = Math.round(nmsWidth / aspect);
  if (nmsCanvas.width !== nmsWidth || nmsCanvas.height !== nmsHeight) {
    nmsCanvas.width = nmsWidth;
    nmsCanvas.height = nmsHeight;
  }

  const sx = cropRect.sx || 0;
  const sy = cropRect.sy || 0;
  const sw = cropRect.sw || video.videoWidth;
  const sh = cropRect.sh || video.videoHeight;
  nmsCtx.drawImage(video, sx, sy, sw, sh, 0, 0, nmsWidth, nmsHeight);
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
  const roiH = Math.floor(h * 0.5);
  const blurSmall = new Float32Array(total);
  const blurLarge = new Float32Array(total);
  if (params.dogMix > 0) {
    boxBlur(brightness, blurSmall, w, h, 1);
    boxBlur(brightness, blurLarge, w, h, 3);
  }
  for (let y = 0; y < roiH; y++) {
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
      const base = brightnessGate * blueGate * satGate;
      const linePenalty = computeLinePenalty(brightness, w, h, x, y);
      mask[idx] = base * linePenalty;
    }
  }

  const points = [];
  const r = nmsConfig.radius;
  for (let y = r; y < roiH - r; y++) {
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

  const stable = stabilizePoints(points);
  const deduped = dedupePoints(stable, 6);
  return { points: deduped, sharpness: computeSharpness(brightness, w, h) };
}

function drawOverlay(points) {
  if (!overlayCtx) return;
  overlayCtx.clearRect(0, 0, overlay.width, overlay.height);

  const sx = viewRect.w / nmsWidth;
  const sy = viewRect.h / nmsHeight;
  overlayCtx.lineWidth = 2;
  overlayCtx.strokeStyle = 'rgba(255, 255, 255, 0.9)';
  overlayCtx.fillStyle = 'rgba(74, 163, 255, 0.85)';

  for (const pt of points) {
    const x = viewRect.x + pt.x * sx;
    const y = viewRect.y + pt.y * sy;
    overlayCtx.beginPath();
    overlayCtx.arc(x, y, 5, 0, Math.PI * 2);
    overlayCtx.stroke();
    overlayCtx.beginPath();
    overlayCtx.arc(x, y, 2.5, 0, Math.PI * 2);
    overlayCtx.fill();
  }
  pointsEl.textContent = `Points: ${points.length}`;
  drawRoiCircle();
  drawReferenceShape();
}

function drawReferenceShape() {
  const pad = 14;
  const boxW = 120;
  const boxH = 180;
  const x0 = overlay.width - boxW - pad;
  const y0 = overlay.height - boxH - pad;

  overlayCtx.save();
  overlayCtx.strokeStyle = 'rgba(255,255,255,0.6)';
  overlayCtx.fillStyle = 'rgba(10,16,24,0.45)';
  overlayCtx.lineWidth = 1.5;
  overlayCtx.beginPath();
  overlayCtx.roundRect(x0, y0, boxW, boxH, 8);
  overlayCtx.fill();
  overlayCtx.stroke();

  const cx = x0 + boxW / 2;
  const cy = y0 + boxH / 2 + 10;
  const scale = 0.7;

  // Draw LED4-LED1 rectangle.
  const rx = 44 * scale;
  const ry = 30 * scale;
  overlayCtx.strokeStyle = 'rgba(74,163,255,0.9)';
  overlayCtx.lineWidth = 2;
  overlayCtx.strokeRect(cx - rx, cy - ry, rx * 2, ry * 2);

  // Draw LED points.
  const leds = [
    { x: cx + rx, y: cy - ry }, // LED1
    { x: cx + rx, y: cy + ry }, // LED2
    { x: cx - rx, y: cy + ry }, // LED3
    { x: cx - rx, y: cy - ry }, // LED4
    { x: cx, y: cy - ry - 50 * scale }, // LED5
  ];
  overlayCtx.fillStyle = 'rgba(255,255,255,0.9)';
  for (const p of leds) {
    overlayCtx.beginPath();
    overlayCtx.arc(p.x, p.y, 4, 0, Math.PI * 2);
    overlayCtx.fill();
  }

  // Draw three horizontal strips.
  overlayCtx.strokeStyle = 'rgba(255,220,120,0.9)';
  overlayCtx.lineWidth = 4;
  for (let i = -1; i <= 1; i++) {
    const y = cy + i * 16 * scale;
    overlayCtx.beginPath();
    overlayCtx.moveTo(cx - 34 * scale, y);
    overlayCtx.lineTo(cx + 34 * scale, y);
    overlayCtx.stroke();
  }

  overlayCtx.restore();
}

function drawRoiCircle() {
  const radius = overlay.width * 0.25;
  const cx = overlay.width / 2;
  const cy = overlay.height * 0.25;
  overlayCtx.save();
  overlayCtx.strokeStyle = 'rgba(0, 200, 255, 0.7)';
  overlayCtx.lineWidth = 2;
  overlayCtx.beginPath();
  overlayCtx.arc(cx, cy, radius, 0, Math.PI * 2);
  overlayCtx.stroke();
  overlayCtx.restore();
}

async function estimatePose(points) {
  await cvReady;
  if (!video.videoWidth || !video.videoHeight) return;

  const candidates = points
    .slice()
    .sort((a, b) => b.score - a.score)
    .slice(0, 12);
  if (candidates.length < 5) {
    updateLock(false);
    scoreEl.textContent = 'Score: -';
    poseEl.textContent = 'Pose: -';
    return;
  }

  const best = findBestPose(candidates);
  if (!best) {
    updateLock(false);
    scoreEl.textContent = 'Score: -';
    poseEl.textContent = 'Pose: -';
    return;
  }

  updateLock(true);
  scoreEl.textContent = `Score: ${best.score.toFixed(3)} err=${best.error.toFixed(2)}`;
  const smooth = applyKalman(best);
  poseEl.textContent =
    `Pose: x=${smooth.tvec[0].toFixed(1)} y=${smooth.tvec[1].toFixed(1)} z=${smooth.tvec[2].toFixed(1)}mm ` +
    `r=${smooth.euler[0].toFixed(1)} p=${smooth.euler[1].toFixed(1)} y=${smooth.euler[2].toFixed(1)}`;
}

function findBestPose(points) {
  const combos = combinations(points, 4);
  let best = null;

  for (const quad of combos) {
    const rect = scoreRectangle(quad);
    if (!rect) continue;
    const led5 = pickLed5(points, rect.center);
    if (!led5) continue;

    const perms = rectanglePermutations(rect.ordered);
    for (const perm of perms) {
      const imgPts = [perm[0], perm[1], perm[2], perm[3], led5];
      const pose = solvePnP(imgPts);
      if (!pose) continue;
      pose.score += rect.score;
      if (!best || pose.error < best.error) best = pose;
    }
  }

  if (best && best.error < 20) return best;
  return null;
}

function scoreRectangle(pts) {
  if (pts.length !== 4) return null;
  const center = meanPoint(pts);
  const ordered = sortByAngle(pts, center);
  const d01 = Math.sqrt(dist2(ordered[0], ordered[1]));
  const d12 = Math.sqrt(dist2(ordered[1], ordered[2]));
  const d23 = Math.sqrt(dist2(ordered[2], ordered[3]));
  const d30 = Math.sqrt(dist2(ordered[3], ordered[0]));
  const w = (d01 + d23) * 0.5;
  const h = (d12 + d30) * 0.5;
  if (w < 2 || h < 2) return null;
  const ratio = w / h;
  const target = 124 / 86.6;
  const ratioErr = Math.abs(Math.log(ratio / target));
  const score = Math.max(0, 1.0 - ratioErr * 1.5);
  if (score < 0.15) return null;
  return { center, ordered, score };
}

function pickLed5(points, center) {
  let best = null;
  let bestScore = -1;
  for (const p of points) {
    const dy = center.y - p.y;
    if (dy < 0) continue;
    const dx = Math.abs(p.x - center.x);
    const score = dy - dx * 0.3;
    if (score > bestScore) {
      bestScore = score;
      best = p;
    }
  }
  if (best) return best;
  // Fallback: pick farthest point from center.
  let maxD = -1;
  for (const p of points) {
    const d = dist2(p, center);
    if (d > maxD) {
      maxD = d;
      best = p;
    }
  }
  return best;
}

function solvePnP(imagePoints) {
  if (imagePoints.length !== 5) return null;
  const w = cropRect.sw || video.videoWidth;
  const h = cropRect.sh || video.videoHeight;
  const fx = 0.9 * Math.max(w, h);
  const fy = fx;
  const cx = w / 2;
  const cy = h / 2;

  const obj = cv.matFromArray(5, 3, cv.CV_64F, objectPoints.flat());
  const img = cv.matFromArray(5, 2, cv.CV_64F, imagePoints.flatMap((p) => [
    cropRect.sx + p.x * (w / nmsWidth),
    cropRect.sy + p.y * (h / nmsHeight),
  ]));
  const cam = cv.matFromArray(3, 3, cv.CV_64F, [
    fx, 0, cx,
    0, fy, cy,
    0, 0, 1,
  ]);
  const dist = cv.Mat.zeros(4, 1, cv.CV_64F);
  const rvec = new cv.Mat();
  const tvec = new cv.Mat();

  const ok = cv.solvePnP(obj, img, cam, dist, rvec, tvec, false, cv.SOLVEPNP_ITERATIVE);
  if (!ok) {
    obj.delete(); img.delete(); cam.delete(); dist.delete(); rvec.delete(); tvec.delete();
    return null;
  }

  const proj = new cv.Mat();
  cv.projectPoints(obj, rvec, tvec, cam, dist, proj);
  let err = 0;
  for (let i = 0; i < 5; i++) {
    const px = proj.data64F[i * 2];
    const py = proj.data64F[i * 2 + 1];
    const ix = img.data64F[i * 2];
    const iy = img.data64F[i * 2 + 1];
    const dx = px - ix;
    const dy = py - iy;
    err += dx * dx + dy * dy;
  }
  err = Math.sqrt(err / 5);

  const r = rvec.data64F;
  const t = tvec.data64F;
  const euler = rvecToEuler(r[0], r[1], r[2]);

  obj.delete(); img.delete(); cam.delete(); dist.delete(); proj.delete(); rvec.delete(); tvec.delete();
  const score = 1 / (1 + err);
  return { error: err, score, rvec: [r[0], r[1], r[2]], tvec: [t[0], t[1], t[2]], euler };
}

function rvecToEuler(rx, ry, rz) {
  const theta = Math.sqrt(rx * rx + ry * ry + rz * rz) || 1e-6;
  const kx = rx / theta;
  const ky = ry / theta;
  const kz = rz / theta;
  const c = Math.cos(theta);
  const s = Math.sin(theta);
  const v = 1 - c;

  const r00 = kx * kx * v + c;
  const r01 = kx * ky * v - kz * s;
  const r02 = kx * kz * v + ky * s;
  const r10 = ky * kx * v + kz * s;
  const r11 = ky * ky * v + c;
  const r12 = ky * kz * v - kx * s;
  const r20 = kz * kx * v - ky * s;
  const r21 = kz * ky * v + kx * s;
  const r22 = kz * kz * v + c;

  const sy = Math.sqrt(r00 * r00 + r10 * r10);
  let roll, pitch, yaw;
  if (sy > 1e-6) {
    roll = Math.atan2(r21, r22);
    pitch = Math.atan2(-r20, sy);
    yaw = Math.atan2(r10, r00);
  } else {
    roll = Math.atan2(-r12, r11);
    pitch = Math.atan2(-r20, sy);
    yaw = 0;
  }
  return [radToDeg(roll), radToDeg(pitch), radToDeg(yaw)];
}

function radToDeg(v) {
  return v * 180 / Math.PI;
}

function updateLock(locked) {
  lockEl.textContent = locked ? 'Tag: LOCK' : 'Tag: searching';
  if (locked && !lastLock) {
    beep();
  }
  lastLock = locked;
}

function computeCropRect(srcW, srcH, dstW, dstH) {
  const srcAspect = srcW / srcH;
  const dstAspect = dstW / dstH;
  let sw = srcW;
  let sh = srcH;
  let sx = 0;
  let sy = 0;
  if (srcAspect > dstAspect) {
    sw = Math.round(srcH * dstAspect);
    sx = Math.round((srcW - sw) / 2);
  } else if (srcAspect < dstAspect) {
    sh = Math.round(srcW / dstAspect);
    sy = Math.round((srcH - sh) / 2);
  }
  return { sx, sy, sw, sh };
}

function applyKalman(pose) {
  if (!filteredPose) {
    filteredPose = {
      tvec: pose.tvec.slice(),
      euler: pose.euler.slice(),
    };
    return filteredPose;
  }
  const a = kalmanConfig.alpha;
  for (let i = 0; i < 3; i++) {
    filteredPose.tvec[i] = filteredPose.tvec[i] * (1 - a) + pose.tvec[i] * a;
    filteredPose.euler[i] = filteredPose.euler[i] * (1 - a) + pose.euler[i] * a;
  }
  return filteredPose;
}

function beep() {
  const now = performance.now();
  if (now - lastBeepTime < 800) return;
  lastBeepTime = now;
  const ctx = new (window.AudioContext || window.webkitAudioContext)();
  const osc = ctx.createOscillator();
  const gain = ctx.createGain();
  osc.type = 'sine';
  osc.frequency.value = 880;
  gain.gain.value = 0.05;
  osc.connect(gain);
  gain.connect(ctx.destination);
  osc.start();
  setTimeout(() => {
    osc.stop();
    ctx.close();
  }, 120);
}

function combinations(arr, k) {
  const result = [];
  const n = arr.length;
  const stack = [];
  const backtrack = (start) => {
    if (stack.length === k) {
      result.push(stack.slice());
      return;
    }
    for (let i = start; i < n; i++) {
      stack.push(arr[i]);
      backtrack(i + 1);
      stack.pop();
    }
  };
  backtrack(0);
  return result;
}

function meanPoint(pts) {
  let sx = 0;
  let sy = 0;
  for (const p of pts) {
    sx += p.x;
    sy += p.y;
  }
  return { x: sx / pts.length, y: sy / pts.length };
}

function dist2(a, b) {
  const dx = a.x - b.x;
  const dy = a.y - b.y;
  return dx * dx + dy * dy;
}

function sortByAngle(pts, center) {
  return pts.slice().sort((a, b) => Math.atan2(a.y - center.y, a.x - center.x) - Math.atan2(b.y - center.y, b.x - center.x));
}

function rectanglePermutations(pts) {
  const perms = [];
  for (let i = 0; i < 4; i++) {
    perms.push([pts[i % 4], pts[(i + 1) % 4], pts[(i + 2) % 4], pts[(i + 3) % 4]]);
  }
  const rev = pts.slice().reverse();
  for (let i = 0; i < 4; i++) {
    perms.push([rev[i % 4], rev[(i + 1) % 4], rev[(i + 2) % 4], rev[(i + 3) % 4]]);
  }
  return perms;
}

function render() {
  if (video.readyState >= 2) {
    gl.bindTexture(gl.TEXTURE_2D, tex);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGB, gl.RGB, gl.UNSIGNED_BYTE, video);
    if (video.videoWidth && video.videoHeight) {
      gl.uniform2f(uniforms.uTexSize, video.videoWidth, video.videoHeight);
      cropRect = computeCropRect(video.videoWidth, video.videoHeight, overlay.width, overlay.height);
      viewRect = { x: 0, y: 0, w: overlay.width, h: overlay.height };
    }
    updateUniforms();
    gl.drawArrays(gl.TRIANGLES, 0, 6);
    const result = computeNmsPoints();
    drawOverlay(result.points);
    sharpnessEl.textContent = `Sharpness: ${result.sharpness.toFixed(3)}`;
    estimatePose(result.points);
    const now = performance.now();
    if (currentStream && now - lastInfoTime > 1000) {
      updateCameraInfo(currentStream);
      lastInfoTime = now;
    }
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
  updateCameraInfo(stream);
  requestAnimationFrame(() => {
    const focusModeSel = document.getElementById('focusMode');
    if (focusModeSel) {
      focusModeSel.dispatchEvent(new Event('change'));
    }
  });
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
  const controls = document.getElementById('controls');
  const toggleBtn = document.getElementById('toggleControls');
  if (controls && toggleBtn) {
    toggleBtn.addEventListener('click', () => {
      const collapsed = controls.classList.toggle('collapsed');
      toggleBtn.textContent = collapsed ? 'Show' : 'Hide';
    });
  }
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

  const focusModeSel = document.getElementById('focusMode');
  const focusModeVal = document.getElementById('focusModeVal');
  const focusDistInput = document.getElementById('focusDistance');
  const focusDistVal = document.getElementById('focusDistanceVal');

  if (focusModeSel && focusModeVal && focusDistInput && focusDistVal) {
    const applyFocus = () => {
      const mode = focusModeSel.value;
      focusModeVal.textContent = mode;

      const track = currentStream ? currentStream.getVideoTracks()[0] : null;
      if (!track) return;

      const dist = Number.parseFloat(focusDistInput.value);
      focusDistVal.textContent = dist.toFixed(3);

      const advanced = [];
      if (mode === 'manual') {
        advanced.push({ focusMode: 'manual', focusDistance: dist });
      } else {
        advanced.push({ focusMode: 'continuous' });
      }

      track.applyConstraints({ advanced }).then(() => {
        updateCameraInfo(currentStream);
      }).catch(() => {});

      localStorage.setItem('focusMode', mode);
      localStorage.setItem('focusDistance', String(dist));
    };

    const updateRange = () => {
      focusDistInput.min = String(focusCaps.min);
      focusDistInput.max = String(focusCaps.max);
      if (Number.isFinite(focusCaps.min) && Number.isFinite(focusCaps.max)) {
        const mid = (focusCaps.min + focusCaps.max) * 0.5;
        focusDistInput.value = String(mid);
        focusDistVal.textContent = mid.toFixed(3);
      }
    };

    focusModeSel.addEventListener('change', applyFocus);
    focusDistInput.addEventListener('input', applyFocus);

    updateRange();
    const savedMode = localStorage.getItem('focusMode');
    const savedDist = localStorage.getItem('focusDistance');
    if (savedMode) {
      focusModeSel.value = savedMode;
      focusModeVal.textContent = savedMode;
    }
    if (savedDist) {
      focusDistInput.value = savedDist;
      focusDistVal.textContent = Number.parseFloat(savedDist).toFixed(3);
    }
  }

  const exposureModeSel = document.getElementById('exposureMode');
  const exposureModeVal = document.getElementById('exposureModeVal');
  const exposureTimeInput = document.getElementById('exposureTime');
  const exposureTimeVal = document.getElementById('exposureTimeVal');
  const isoInput = document.getElementById('iso');
  const isoVal = document.getElementById('isoVal');
  const exposureCompInput = document.getElementById('exposureComp');
  const exposureCompVal = document.getElementById('exposureCompVal');

  if (exposureModeSel && exposureModeVal && exposureTimeInput && exposureTimeVal && isoInput && isoVal && exposureCompInput && exposureCompVal) {
    const updateRanges = () => {
      exposureTimeInput.min = String(exposureCaps.min);
      exposureTimeInput.max = String(exposureCaps.max);
      isoInput.min = String(isoCaps.min);
      isoInput.max = String(isoCaps.max);
      exposureCompInput.min = String(expCompCaps.min);
      exposureCompInput.max = String(expCompCaps.max);
    };

    const applyExposure = () => {
      const mode = exposureModeSel.value;
      exposureModeVal.textContent = mode;

      const track = currentStream ? currentStream.getVideoTracks()[0] : null;
      if (!track) return;

      const exposureTime = Number.parseFloat(exposureTimeInput.value);
      const iso = Number.parseFloat(isoInput.value);
      const expComp = Number.parseFloat(exposureCompInput.value);

      exposureTimeVal.textContent = exposureTime.toFixed(1);
      isoVal.textContent = Math.round(iso);
      exposureCompVal.textContent = expComp.toFixed(1);

      const advanced = [];
      if (mode === 'manual') {
        advanced.push({
          exposureMode: 'manual',
          exposureTime,
          iso,
          exposureCompensation: expComp,
        });
      } else {
        advanced.push({ exposureMode: 'continuous' });
      }

      track.applyConstraints({ advanced }).then(() => {
        updateCameraInfo(currentStream);
      }).catch(() => {});

      localStorage.setItem('exposureMode', mode);
      localStorage.setItem('exposureTime', String(exposureTime));
      localStorage.setItem('iso', String(iso));
      localStorage.setItem('exposureComp', String(expComp));
    };

    exposureModeSel.addEventListener('change', applyExposure);
    exposureTimeInput.addEventListener('input', applyExposure);
    isoInput.addEventListener('input', applyExposure);
    exposureCompInput.addEventListener('input', applyExposure);

    updateRanges();
    const savedMode = localStorage.getItem('exposureMode');
    const savedTime = localStorage.getItem('exposureTime');
    const savedIso = localStorage.getItem('iso');
    const savedComp = localStorage.getItem('exposureComp');
    if (savedMode) {
      exposureModeSel.value = savedMode;
      exposureModeVal.textContent = savedMode;
    }
    if (savedTime) {
      exposureTimeInput.value = savedTime;
      exposureTimeVal.textContent = Number.parseFloat(savedTime).toFixed(1);
    }
    if (savedIso) {
      isoInput.value = savedIso;
      isoVal.textContent = String(Math.round(Number.parseFloat(savedIso)));
    }
    if (savedComp) {
      exposureCompInput.value = savedComp;
      exposureCompVal.textContent = Number.parseFloat(savedComp).toFixed(1);
    }
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

function computeLinePenalty(brightness, w, h, x, y) {
  // Penalize points that look like part of a long horizontal strip.
  const idx = y * w + x;
  const v = brightness[idx];
  if (v < 0.2) return 1.0;

  let left = 0;
  let right = 0;
  for (let i = 1; i <= 12; i++) {
    if (x - i < 0) break;
    if (brightness[idx - i] < v * 0.5) break;
    left++;
  }
  for (let i = 1; i <= 12; i++) {
    if (x + i >= w) break;
    if (brightness[idx + i] < v * 0.5) break;
    right++;
  }
  const run = left + right + 1;
  if (run >= 10) return 0.6;
  if (run >= 7) return 0.8;
  return 1.0;
}

function stabilizePoints(points) {
  if (prevPoints.length === 0) {
    prevPoints = points;
    return points;
  }
  const matched = [];
  for (const p of points) {
    let best = null;
    let bestD = 9;
    for (const q of prevPoints) {
      const d = dist2(p, q);
      if (d < bestD) {
        bestD = d;
        best = q;
      }
    }
    if (best) {
      matched.push({
        x: p.x * 0.8 + best.x * 0.2,
        y: p.y * 0.8 + best.y * 0.2,
        score: p.score,
      });
    } else {
      matched.push(p);
    }
  }
  prevPoints = matched;
  return matched;
}

function dedupePoints(points, minDist) {
  const result = [];
  const minDist2 = minDist * minDist;
  const sorted = points.slice().sort((a, b) => b.score - a.score);
  for (const p of sorted) {
    let keep = true;
    for (const q of result) {
      if (dist2(p, q) < minDist2) {
        keep = false;
        break;
      }
    }
    if (keep) result.push(p);
  }
  return result;
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

function updateCameraInfo(stream) {
  const track = stream.getVideoTracks()[0];
  if (!track) return;
  const caps = track.getCapabilities ? track.getCapabilities() : {};
  const settings = track.getSettings ? track.getSettings() : {};
  if (caps.focusDistance) {
    focusCaps = { min: caps.focusDistance.min ?? 0, max: caps.focusDistance.max ?? 1 };
  }
  if (caps.exposureTime) {
    exposureCaps = { min: caps.exposureTime.min ?? 1, max: caps.exposureTime.max ?? 33 };
  }
  if (caps.iso) {
    isoCaps = { min: caps.iso.min ?? 50, max: caps.iso.max ?? 800 };
  }
  if (caps.exposureCompensation) {
    expCompCaps = {
      min: caps.exposureCompensation.min ?? -2,
      max: caps.exposureCompensation.max ?? 2,
    };
  }

  void settings;
}
