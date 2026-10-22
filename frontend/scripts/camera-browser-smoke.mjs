import { mkdtemp, rm } from "node:fs/promises";
import { spawn } from "node:child_process";
import { tmpdir } from "node:os";
import { join } from "node:path";

const chrome = process.env.CHROME_PATH ?? "C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe";
const port = 9333;
const useRealCamera = process.env.SUREINSPECT_REAL_CAMERA === "1" || process.argv.includes("--real-camera");
const profile = await mkdtemp(join(tmpdir(), "sureinspect-chrome-"));
const chromeArgs = [
  `--remote-debugging-port=${port}`,
  `--user-data-dir=${profile}`,
  "--use-fake-ui-for-media-stream",
  "--no-first-run",
  "--disable-default-apps",
  "http://127.0.0.1:3000",
];
if (useRealCamera) {
  // Real Windows webcams are not exposed to Chrome's headless media stack.
  // Keep the temporary test window unobtrusive while using the normal stack.
  chromeArgs.unshift("--start-minimized", "--disable-backgrounding-occluded-windows");
} else {
  chromeArgs.unshift("--headless=new", "--use-fake-device-for-media-stream");
}
const browser = spawn(chrome, chromeArgs, { stdio: "ignore" });

const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));
let socket;
let nextId = 1;
const pending = new Map();
const browserErrors = [];

async function targets() {
  for (let attempt = 0; attempt < 40; attempt += 1) {
    try {
      const response = await fetch(`http://127.0.0.1:${port}/json/list`);
      if (response.ok) return response.json();
    } catch { /* browser still starting */ }
    await sleep(250);
  }
  throw new Error("Chrome DevTools endpoint did not start");
}

function command(method, params = {}) {
  const id = nextId++;
  return new Promise((resolve, reject) => {
    pending.set(id, { resolve, reject });
    socket.send(JSON.stringify({ id, method, params }));
  });
}

async function evaluate(expression) {
  const response = await command("Runtime.evaluate", { expression, awaitPromise: true, returnByValue: true });
  if (response.exceptionDetails) throw new Error(response.exceptionDetails.text);
  return response.result.value;
}

async function waitFor(expression, label, timeoutMs = 8000) {
  const deadline = Date.now() + timeoutMs;
  while (Date.now() < deadline) {
    if (await evaluate(expression)) return;
    await sleep(100);
  }
  const visibleText = await evaluate("document.body.innerText.slice(0, 1600)");
  throw new Error(`Timed out waiting for ${label}. Visible page text:\n${visibleText}`);
}

async function clickButton(label) {
  const clicked = await evaluate(`(() => {
    const button = [...document.querySelectorAll('button')].find((item) => item.textContent?.trim().includes(${JSON.stringify(label)}));
    if (!button || button.disabled) return false;
    button.click();
    return true;
  })()`);
  if (!clicked) throw new Error(`Button not found or disabled: ${label}`);
}

try {
  const list = await targets();
  const page = list.find((item) => item.type === "page" && item.url.includes("127.0.0.1:3000"));
  if (!page) throw new Error("SureInspect page target not found");
  socket = new WebSocket(page.webSocketDebuggerUrl);
  await new Promise((resolve, reject) => {
    socket.addEventListener("open", resolve, { once: true });
    socket.addEventListener("error", reject, { once: true });
    socket.addEventListener("message", (event) => {
      const message = JSON.parse(event.data);
      if (message.method === "Runtime.exceptionThrown") browserErrors.push(message.params.exceptionDetails.text);
      if (message.method === "Log.entryAdded" && message.params.entry.level === "error") {
        const entry = message.params.entry;
        if (!entry.url?.endsWith("/favicon.ico")) browserErrors.push(`${entry.text}${entry.url ? ` (${entry.url})` : ""}`);
      }
      if (!message.id || !pending.has(message.id)) return;
      const job = pending.get(message.id);
      pending.delete(message.id);
      if (message.error) job.reject(new Error(message.error.message));
      else job.resolve(message.result);
    });
  });
  await command("Runtime.enable");
  await command("Log.enable");
  await command("Emulation.setDeviceMetricsOverride", { width: 390, height: 844, deviceScaleFactor: 1, mobile: true });
  await command("Page.reload", { ignoreCache: true });
  await waitFor("document.querySelector('button')?.textContent !== undefined", "application render");

  await clickButton("Live Camera");
  await waitFor("document.querySelector('[role=dialog]') !== null", "camera dialog");
  await waitFor("[...document.querySelectorAll('button')].some(b => b.title === 'Capture Still Frame' && !b.disabled)", "hardware camera stream");
  await sleep(500);
  const cameraTelemetry = await evaluate(`(() => {
    const video = document.querySelector('video');
    const track = video?.srcObject?.getVideoTracks?.()[0];
    if (!video || !track || !video.videoWidth || !video.videoHeight) return null;
    const canvas = document.createElement('canvas');
    canvas.width = 160;
    canvas.height = 120;
    const context = canvas.getContext('2d', { willReadFrequently: true });
    context.drawImage(video, 0, 0, canvas.width, canvas.height);
    const pixels = context.getImageData(0, 0, canvas.width, canvas.height).data;
    let total = 0;
    let totalSquared = 0;
    let nonBlack = 0;
    const count = pixels.length / 4;
    for (let index = 0; index < pixels.length; index += 4) {
      const luminance = (pixels[index] + pixels[index + 1] + pixels[index + 2]) / 3;
      total += luminance;
      totalSquared += luminance * luminance;
      if (luminance > 3) nonBlack += 1;
    }
    const mean = total / count;
    return {
      label: track.label,
      width: video.videoWidth,
      height: video.videoHeight,
      readyState: video.readyState,
      meanLuminance: Number(mean.toFixed(2)),
      luminanceStdDev: Number(Math.sqrt(Math.max(0, totalSquared / count - mean * mean)).toFixed(2)),
      nonBlackRatio: Number((nonBlack / count).toFixed(4))
    };
  })()`);
  if (!cameraTelemetry) throw new Error("Camera stream opened without a measurable video frame");
  if (useRealCamera && cameraTelemetry.nonBlackRatio < 0.01) {
    throw new Error(`Physical camera is producing a black frame: ${JSON.stringify(cameraTelemetry)}`);
  }
  await evaluate("document.querySelector('button[title=\"Capture Still Frame\"]').click()");
  await waitFor("document.body.innerText.includes('CAMERA FRAME CAPTURED')", "hardware preview");
  await clickButton("Retake");
  await waitFor("[...document.querySelectorAll('button')].some(b => b.title === 'Capture Still Frame' && !b.disabled)", "retaken camera stream");
  await evaluate("document.dispatchEvent(new KeyboardEvent('keydown', {key: 'Escape', bubbles: true}))");
  await waitFor("document.querySelector('[role=dialog]') === null", "Escape close");

  await clickButton("Live Camera");
  await waitFor("document.querySelector('[role=dialog]') !== null", "reopened camera dialog");
  await clickButton("Demo Simulation");
  await waitFor("document.body.innerText.includes('Demo simulation active')", "simulation disclosure");
  await evaluate("document.querySelector('button[title=\"Capture Still Frame\"]').click()");
  await waitFor("document.body.innerText.includes('SIMULATED DEMO FRAME')", "simulated preview");
  await clickButton("Use Demo Frame");
  await waitFor("document.body.innerText.includes('simulated_demo_')", "attached simulated filename");

  const result = await evaluate(`({
    title: document.title,
    viewport: [window.innerWidth, window.innerHeight],
    dialogClosed: document.querySelector('[role=dialog]') === null,
    simulatedFileAttached: document.body.innerText.includes('simulated_demo_'),
    bodyOverflowRestored: document.body.style.overflow === ''
  })`);
  if (browserErrors.length) throw new Error(`Browser errors observed: ${browserErrors.join(" | ")}`);
  console.log(JSON.stringify({ passed: true, cameraMode: useRealCamera ? "physical" : "fake", cameraTelemetry, ...result }, null, 2));
} finally {
  if (socket?.readyState === WebSocket.OPEN) socket.close();
  if (browser.exitCode === null && browser.signalCode === null) {
    const exited = new Promise((resolve) => browser.once("exit", resolve));
    browser.kill();
    await Promise.race([exited, sleep(2_000)]);
  }
  await rm(profile, { recursive: true, force: true });
}
