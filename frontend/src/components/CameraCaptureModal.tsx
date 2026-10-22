import { useCallback, useEffect, useRef, useState } from "react";
import { Camera, Check, RefreshCw, X, AlertCircle, Video } from "lucide-react";

const CAMERA_READY_TIMEOUT_MS = 8_000;

function cameraPreferenceScore(device: MediaDeviceInfo) {
  const label = device.label.toLowerCase();
  let score = 0;
  if (/front|user|integrated|built.?in|webcam/.test(label)) score += 20;
  if (/camera|sonix/.test(label)) score += 5;
  if (/rear|back|environment/.test(label)) score -= 20;
  if (/virtual|obs|snap camera/.test(label)) score -= 40;
  return score;
}

function preferredCamera(devices: MediaDeviceInfo[]) {
  return [...devices].sort((left, right) => cameraPreferenceScore(right) - cameraPreferenceScore(left))[0];
}

function waitForVideoFrame(video: HTMLVideoElement) {
  if (video.readyState >= HTMLMediaElement.HAVE_CURRENT_DATA && video.videoWidth > 0) {
    return Promise.resolve();
  }

  return new Promise<void>((resolve, reject) => {
    const timeout = window.setTimeout(() => {
      cleanup();
      reject(new Error("Camera opened, but no video frames arrived"));
    }, CAMERA_READY_TIMEOUT_MS);

    const onReady = () => {
      if (video.videoWidth <= 0) return;
      cleanup();
      resolve();
    };
    const onError = () => {
      cleanup();
      reject(new Error("The browser could not display the camera stream"));
    };
    const cleanup = () => {
      window.clearTimeout(timeout);
      video.removeEventListener("loadeddata", onReady);
      video.removeEventListener("canplay", onReady);
      video.removeEventListener("error", onError);
    };

    video.addEventListener("loadeddata", onReady);
    video.addEventListener("canplay", onReady);
    video.addEventListener("error", onError);
  });
}

async function videoFrameAppearsBlack(video: HTMLVideoElement) {
  await new Promise((resolve) => window.setTimeout(resolve, 400));
  const canvas = document.createElement("canvas");
  canvas.width = 64;
  canvas.height = 48;
  const context = canvas.getContext("2d", { willReadFrequently: true });
  if (!context) return false;
  context.drawImage(video, 0, 0, canvas.width, canvas.height);
  const pixels = context.getImageData(0, 0, canvas.width, canvas.height).data;
  let visiblePixels = 0;
  for (let index = 0; index < pixels.length; index += 4) {
    if (pixels[index] + pixels[index + 1] + pixels[index + 2] > 12) visiblePixels += 1;
  }
  return visiblePixels / (pixels.length / 4) < 0.01;
}

interface Props {
  isOpen: boolean;
  onClose: () => void;
  onCapture: (file: File) => void;
}

export default function CameraCaptureModal({ isOpen, onClose, onCapture }: Props) {
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const simCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const closeButtonRef = useRef<HTMLButtonElement | null>(null);
  const previousFocusRef = useRef<HTMLElement | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const animFrameRef = useRef<number | null>(null);
  const cameraAttemptRef = useRef(0);

  const [devices, setDevices] = useState<MediaDeviceInfo[]>([]);
  const [selectedDeviceId, setSelectedDeviceId] = useState<string>("");
  const [streaming, setStreaming] = useState(false);
  const [requesting, setRequesting] = useState(false);
  const [isSimulated, setIsSimulated] = useState(false);
  const [activeCameraLabel, setActiveCameraLabel] = useState<string>("");
  const [error, setError] = useState<string | null>(null);
  const [previewBlob, setPreviewBlob] = useState<Blob | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);

  // Stop active media stream tracks
  const stopStream = useCallback(() => {
    cameraAttemptRef.current += 1;
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((track) => track.stop());
      streamRef.current = null;
    }
    if (animFrameRef.current) {
      cancelAnimationFrame(animFrameRef.current);
      animFrameRef.current = null;
    }
    setStreaming(false);
    setActiveCameraLabel("");
  }, []);

  // Enumerate video devices
  const updateDeviceList = useCallback(async () => {
    try {
      if (!navigator.mediaDevices?.enumerateDevices) return;
      const all = await navigator.mediaDevices.enumerateDevices();
      const videoDevs = all.filter((d) => d.kind === "videoinput");
      setDevices(videoDevs);
      if (videoDevs.length > 0 && !selectedDeviceId) {
        setSelectedDeviceId(preferredCamera(videoDevs).deviceId);
      }
    } catch {
      // Ignore enumeration errors
    }
  }, [selectedDeviceId]);

  // Start simulated optical inspection feed (fallback guaranteed to always work)
  const startSimulatedFeed = useCallback(() => {
    stopStream();
    setError("Demo simulation active — this is generated imagery, not a physical camera feed.");
    setIsSimulated(true);
    setStreaming(true);
    setRequesting(false);

    const canvas = simCanvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    canvas.width = 800;
    canvas.height = 600;

    let tick = 0;
    const renderLoop = () => {
      tick++;
      // Background optical bench surface
      ctx.fillStyle = "#1e2220";
      ctx.fillRect(0, 0, canvas.width, canvas.height);

      // Subtle conveyor belt texture lines
      ctx.strokeStyle = "#2a302c";
      ctx.lineWidth = 1;
      const offset = (tick * 1.5) % 40;
      for (let y = offset; y < canvas.height; y += 40) {
        ctx.beginPath();
        ctx.moveTo(0, y);
        ctx.lineTo(canvas.width, y);
        ctx.stroke();
      }

      // Realistic PCB board representation
      const pcbX = 160 + Math.sin(tick * 0.02) * 5;
      const pcbY = 80 + Math.cos(tick * 0.02) * 4;
      const pcbW = 480;
      const pcbH = 440;

      // PCB substrate (green FR-4)
      ctx.fillStyle = "#175b3c";
      ctx.beginPath();
      ctx.roundRect(pcbX, pcbY, pcbW, pcbH, 12);
      ctx.fill();
      ctx.strokeStyle = "#258459";
      ctx.lineWidth = 3;
      ctx.stroke();

      // Copper trace lines
      ctx.strokeStyle = "#c59b27";
      ctx.lineWidth = 2;
      for (let i = 0; i < 6; i++) {
        const ty = pcbY + 60 + i * 50;
        ctx.beginPath();
        ctx.moveTo(pcbX + 40, ty);
        ctx.lineTo(pcbX + 180, ty);
        ctx.lineTo(pcbX + 220, ty + 20);
        ctx.lineTo(pcbX + 420, ty + 20);
        ctx.stroke();
      }

      // Integrated Circuit (QFP Chip)
      const icX = pcbX + 200;
      const icY = pcbY + 160;
      const icW = 120;
      const icH = 120;
      ctx.fillStyle = "#121413";
      ctx.fillRect(icX, icY, icW, icH);
      ctx.strokeStyle = "#383d3b";
      ctx.strokeRect(icX, icY, icW, icH);

      // IC Pins
      ctx.fillStyle = "#d1d5db";
      for (let p = 0; p < 8; p++) {
        ctx.fillRect(icX - 10, icY + 15 + p * 12, 10, 4); // left pins
        ctx.fillRect(icX + icW, icY + 15 + p * 12, 10, 4); // right pins
        ctx.fillRect(icX + 15 + p * 12, icY - 10, 4, 10); // top pins
        ctx.fillRect(icX + 15 + p * 12, icY + icH, 4, 10); // bottom pins
      }

      // Chip label text
      ctx.fillStyle = "#9ca3af";
      ctx.font = "bold 11px monospace";
      ctx.fillText("STM32F4", icX + 28, icY + 65);

      // Gold surface-mount test pads
      ctx.fillStyle = "#facc15";
      for (let c = 0; c < 5; c++) {
        for (let r = 0; r < 3; r++) {
          ctx.beginPath();
          ctx.arc(pcbX + 70 + c * 30, pcbY + 360 + r * 20, 4, 0, Math.PI * 2);
          ctx.fill();
        }
      }

      animFrameRef.current = requestAnimationFrame(renderLoop);
    };

    renderLoop();
  }, [stopStream]);

  // Start hardware camera stream with multi-device fallback
  const startCamera = useCallback(async (deviceIdOverride?: string) => {
    stopStream();
    const attempt = cameraAttemptRef.current;
    setError(null);
    setIsSimulated(false);
    setRequesting(true);
    setPreviewBlob(null);
    setPreviewUrl((prev) => {
      if (prev) URL.revokeObjectURL(prev);
      return null;
    });

    // Browsers may release a stopped hardware track asynchronously. A brief
    // handoff prevents an immediate Retake from racing the previous stream.
    await new Promise((resolve) => setTimeout(resolve, 200));
    if (attempt !== cameraAttemptRef.current) return;

    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      setError("Camera access is not supported by your browser or requires a secure context (HTTPS/localhost).");
      setRequesting(false);
      return;
    }

    const devId = deviceIdOverride || selectedDeviceId;
    let stream: MediaStream | null = null;
    let lastCameraError: unknown = null;

    try {
      // 1. Try with chosen deviceId if specified
      if (devId) {
        try {
          stream = await navigator.mediaDevices.getUserMedia({
            video: { deviceId: { exact: devId } },
            audio: false,
          });
        } catch (cameraError) {
          lastCameraError = cameraError;
          // Fall through
        }
      }

      // 2. Prefer the front/user-facing camera. This is the correct default
      // for laptop webcams; phones may still choose a rear camera explicitly.
      if (!stream) {
        try {
          stream = await navigator.mediaDevices.getUserMedia({
            video: {
              facingMode: { ideal: "user" },
              width: { ideal: 1280 },
              height: { ideal: 720 },
            },
            audio: false,
          });
        } catch (cameraError) {
          lastCameraError = cameraError;
          // Fall through
        }
      }

      // 3. Multi-device scan fallback: try all enumerated video devices
      if (!stream && navigator.mediaDevices.enumerateDevices) {
        try {
          const allDevs = await navigator.mediaDevices.enumerateDevices();
          const videoDevs = allDevs.filter((d) => d.kind === "videoinput");
          setDevices(videoDevs);

          const candidates = [...videoDevs].sort(
            (left, right) => cameraPreferenceScore(right) - cameraPreferenceScore(left)
          );
          for (const d of candidates) {
            if (d.deviceId === devId) continue;
            try {
              stream = await navigator.mediaDevices.getUserMedia({
                video: { deviceId: { exact: d.deviceId } },
                audio: false,
              });
              if (stream) {
                setSelectedDeviceId(d.deviceId);
                break;
              }
            } catch (cameraError) {
              lastCameraError = cameraError;
              // Try next device
            }
          }
        } catch {
          // Fall through
        }
      }

      if (stream) {
        if (attempt !== cameraAttemptRef.current) {
          stream.getTracks().forEach((track) => track.stop());
          return;
        }
        streamRef.current = stream;
        if (videoRef.current) {
          videoRef.current.srcObject = stream;
          await videoRef.current.play();
          await waitForVideoFrame(videoRef.current);
          if (await videoFrameAppearsBlack(videoRef.current)) {
            setError(
              "The camera opened, but its frames are black. Check the laptop's physical privacy shutter or camera hotkey, then retry."
            );
          }
        } else {
          throw new Error("Camera viewfinder is unavailable");
        }
        const activeTrack = stream.getVideoTracks()[0];
        const activeSettings = activeTrack?.getSettings();
        if (activeSettings?.deviceId) setSelectedDeviceId(activeSettings.deviceId);
        setActiveCameraLabel(activeTrack?.label || "Physical camera");
        setStreaming(true);
        setRequesting(false);
        void updateDeviceList();
      } else {
        if (lastCameraError) throw lastCameraError;
        setRequesting(false);
        setError("No physical camera stream was returned. Retry it, choose a file, or explicitly open the labeled demo simulation.");
      }
    } catch (err: unknown) {
      if (attempt !== cameraAttemptRef.current) return;
      if (stream) stream.getTracks().forEach((track) => track.stop());
      if (streamRef.current === stream) streamRef.current = null;
      const msg = err instanceof Error ? err.message : String(err);
      const errorName = err instanceof DOMException ? err.name : "";
      setRequesting(false);
      setStreaming(false);
      if (errorName === "NotAllowedError" || msg.includes("Permission") || msg.includes("NotAllowed")) {
        setError("Camera permission was denied. Allow camera access for this site, then retry.");
      } else if (errorName === "NotReadableError" || msg.includes("NotReadable") || msg.includes("Could not start")) {
        setError("The camera is being used by another application or is unavailable. Close the other application, then retry.");
      } else if (errorName === "NotFoundError" || msg.includes("NotFound") || msg.includes("DevicesNotFound")) {
        setError("No physical camera was detected. Choose an image file or use the explicitly labeled demo simulation.");
      } else if (errorName === "OverconstrainedError") {
        setError("The selected camera cannot satisfy the requested video settings. Choose another camera and retry.");
      } else if (errorName === "AbortError") {
        setError("Windows started the camera but interrupted it before video arrived. Close camera-using apps and retry.");
      } else if (errorName === "SecurityError") {
        setError("The browser blocked camera access for this page. Open SureInspect on HTTPS or localhost and allow camera access.");
      } else {
        setError(`Unable to start the physical camera${errorName ? ` (${errorName})` : ""}: ${msg}`);
      }
    }
  }, [selectedDeviceId, stopStream, updateDeviceList]);

  // Handle open/close lifecycle
  useEffect(() => {
    if (isOpen) {
      previousFocusRef.current = document.activeElement instanceof HTMLElement ? document.activeElement : null;
      const previousOverflow = document.body.style.overflow;
      document.body.style.overflow = "hidden";
      const focusFrame = requestAnimationFrame(() => closeButtonRef.current?.focus());
      const onKeyDown = (event: KeyboardEvent) => {
        if (event.key === "Escape") onClose();
      };
      document.addEventListener("keydown", onKeyDown);
      void updateDeviceList();
      void startCamera();
      return () => {
        cancelAnimationFrame(focusFrame);
        document.removeEventListener("keydown", onKeyDown);
        document.body.style.overflow = previousOverflow;
        stopStream();
        previousFocusRef.current?.focus();
      };
    } else {
      stopStream();
      setPreviewUrl((prev) => {
        if (prev) URL.revokeObjectURL(prev);
        return null;
      });
      setPreviewBlob(null);
      setIsSimulated(false);
    }
  }, [isOpen]); // eslint-disable-line react-hooks/exhaustive-deps

  useEffect(() => {
    if (!isOpen || !navigator.mediaDevices?.addEventListener) return;
    const handleDeviceChange = () => void updateDeviceList();
    navigator.mediaDevices.addEventListener("devicechange", handleDeviceChange);
    return () => navigator.mediaDevices.removeEventListener("devicechange", handleDeviceChange);
  }, [isOpen, updateDeviceList]);

  // Take snapshot from active source (real video or simulated canvas)
  const takeSnapshot = () => {
    const canvas = document.createElement("canvas");
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    if (isSimulated && simCanvasRef.current) {
      canvas.width = simCanvasRef.current.width;
      canvas.height = simCanvasRef.current.height;
      ctx.drawImage(simCanvasRef.current, 0, 0);
    } else if (videoRef.current && streaming && videoRef.current.readyState >= HTMLMediaElement.HAVE_CURRENT_DATA) {
      const video = videoRef.current;
      canvas.width = video.videoWidth || 1280;
      canvas.height = video.videoHeight || 720;
      ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
    } else {
      return;
    }

    canvas.toBlob(
      (blob) => {
        if (!blob) return;
        const url = URL.createObjectURL(blob);
        if (previewUrl) URL.revokeObjectURL(previewUrl);
        setPreviewBlob(blob);
        setPreviewUrl(url);
        stopStream();
      },
      "image/jpeg",
      0.92
    );
  };

  // Confirm snapshot and return File to parent
  const handleConfirm = () => {
    if (!previewBlob) return;
    const filename = `${isSimulated ? "simulated_demo" : "camera"}_${Date.now()}.jpg`;
    const file = new File([previewBlob], filename, { type: "image/jpeg" });
    onCapture(file);
    onClose();
  };

  // Retake photo
  const handleRetake = () => {
    if (previewUrl) {
      URL.revokeObjectURL(previewUrl);
      setPreviewUrl(null);
    }
    setPreviewBlob(null);
    if (isSimulated) {
      startSimulatedFeed();
    } else {
      void startCamera();
    }
  };

  // Fallback direct file input
  const handleFallbackFile = (e: React.ChangeEvent<HTMLInputElement>) => {
    const f = e.target.files?.[0];
    if (f) {
      onCapture(f);
      onClose();
    }
  };

  if (!isOpen) return null;

  return (
    <div
      role="dialog"
      aria-modal="true"
      aria-labelledby="camera-dialog-title"
      aria-describedby="camera-dialog-description"
      onMouseDown={(event) => { if (event.target === event.currentTarget) onClose(); }}
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/80 backdrop-blur-xs p-3 sm:p-6 camera-safe-area"
    >
      <div className="relative flex flex-col w-full max-w-xl max-h-[92dvh] rounded-2xl bg-[#1b1d1f] text-white shadow-2xl overflow-hidden border border-white/10">
        {/* Header */}
        <div className="flex items-center justify-between px-4 sm:px-5 py-3 border-b border-white/10 bg-white/5">
          <div className="flex items-center gap-2">
            <Camera aria-hidden="true" size={18} className="text-[#34d399]" />
            <span id="camera-dialog-title" className="font-medium text-xs sm:text-sm">
              {isSimulated ? "Demo Camera Simulation" : "Camera Viewfinder"}
            </span>
          </div>

          <div className="flex items-center gap-2">
            {/* Camera device selector if multiple exist */}
            {!isSimulated && devices.length > 1 && !previewBlob && (
              <select
                className="bg-black/60 border border-white/20 text-white text-[11px] rounded-md px-2 py-1 max-w-[140px] truncate"
                value={selectedDeviceId}
                onChange={(e) => {
                  setSelectedDeviceId(e.target.value);
                  void startCamera(e.target.value);
                }}
              >
                {devices.map((d, idx) => (
                  <option key={d.deviceId || idx} value={d.deviceId}>
                    {d.label || `Camera ${idx + 1}`}
                  </option>
                ))}
              </select>
            )}

            <button
              ref={closeButtonRef}
              type="button"
              onClick={onClose}
              title="Close Camera"
              aria-label="Close camera"
              className="p-1.5 rounded-lg bg-white/10 hover:bg-white/20 text-white transition-colors cursor-pointer"
            >
              <X aria-hidden="true" size={16} />
            </button>
          </div>
        </div>

        {/* Informational banner if running on simulated feed */}
        {error && (
          <div className="px-4 py-2 bg-amber-500/10 border-b border-amber-500/20 text-amber-200 text-xs flex items-center gap-2">
            <AlertCircle aria-hidden="true" size={14} className="shrink-0 text-amber-400" />
            <span role="status">{error}</span>
          </div>
        )}

        {/* Viewfinder / Review Area */}
        <div className="relative flex-1 min-h-[300px] sm:min-h-[400px] bg-black flex items-center justify-center overflow-hidden">
          {previewUrl ? (
            <div className="relative w-full h-full flex items-center justify-center bg-black">
              <img
                src={previewUrl}
                alt={isSimulated ? "Captured generated demo frame" : "Captured camera frame"}
                className="max-h-[60dvh] w-auto max-w-full object-contain"
              />
              <div className="absolute top-3 left-3 bg-black/60 px-3 py-1 rounded-full text-xs font-mono tracking-wider text-[#34d399] border border-[#34d399]/30">
                {isSimulated ? "SIMULATED DEMO FRAME" : "CAMERA FRAME CAPTURED"}
              </div>
            </div>
          ) : (
            <div className="relative w-full h-full flex items-center justify-center">
              {requesting && (
                <div role="status" className="absolute inset-0 z-20 flex flex-col items-center justify-center gap-3 bg-black text-white/80">
                  <RefreshCw aria-hidden="true" size={26} className="animate-spin" />
                  <p className="text-sm">Requesting camera access…</p>
                </div>
              )}
              {/* Hardware Video Stream */}
              <video
                ref={videoRef}
                playsInline
                autoPlay
                muted
                className={`w-full h-full object-contain bg-black ${isSimulated ? "hidden" : "block"}`}
              />

              {/* Simulated Optical Canvas Feed */}
              <canvas
                ref={simCanvasRef}
                className={`w-full h-full object-contain bg-black ${isSimulated ? "block" : "hidden"}`}
              />

              {/* Optical Reticle & Alignment Guide */}
              <div className="pointer-events-none absolute inset-6 sm:inset-10 border-2 border-white/30 rounded-xl flex items-center justify-center">
                <div className="w-10 h-10 border border-white/40 rounded-full flex items-center justify-center">
                  <div className="w-1.5 h-1.5 bg-[#34d399] rounded-full animate-ping" />
                </div>
                {/* Corner Accents */}
                <div className="absolute -top-1 -left-1 w-4 h-4 border-t-2 border-l-2 border-[#34d399]" />
                <div className="absolute -top-1 -right-1 w-4 h-4 border-t-2 border-r-2 border-[#34d399]" />
                <div className="absolute -bottom-1 -left-1 w-4 h-4 border-b-2 border-l-2 border-[#34d399]" />
                <div className="absolute -bottom-1 -right-1 w-4 h-4 border-b-2 border-r-2 border-[#34d399]" />
              </div>

              <div className="absolute bottom-3 left-1/2 -translate-x-1/2 bg-black/60 backdrop-blur-md px-3 py-1 rounded-full text-[11px] text-white/80 border border-white/10 whitespace-nowrap">
                {isSimulated
                  ? "Generated demo — not a physical camera"
                  : activeCameraLabel
                    ? `${activeCameraLabel} — center the target within the frame`
                    : "Center the inspection target within the frame"}
              </div>
            </div>
          )}
        </div>

        {/* Footer Action Bar */}
        <div className="px-5 py-3.5 border-t border-white/10 bg-white/5 flex items-center justify-between gap-4">
          {previewBlob ? (
            <div className="flex w-full gap-3">
              <button
                type="button"
                onClick={handleRetake}
                className="flex-1 py-2.5 px-4 rounded-xl border border-white/20 text-white font-medium text-xs sm:text-sm hover:bg-white/10 active:scale-98 transition cursor-pointer"
              >
                Retake
              </button>
              <button
                type="button"
                onClick={handleConfirm}
                className="flex-1 py-2.5 px-4 rounded-xl bg-[#0e5a4c] text-white font-medium text-xs sm:text-sm hover:bg-[#127260] active:scale-98 transition flex items-center justify-center gap-2 cursor-pointer shadow-xs"
              >
                <Check aria-hidden="true" size={16} />
                <span>{isSimulated ? "Use Demo Frame" : "Use This Photo"}</span>
              </button>
            </div>
          ) : (
            <div className="flex items-center justify-between w-full">
              {/* Option to upload file directly */}
              <label className="flex items-center gap-1.5 text-xs text-white/70 hover:text-white cursor-pointer transition">
                <Camera aria-hidden="true" size={14} />
                <span>Upload file</span>
                <input
                  type="file"
                  accept="image/png,image/jpeg"
                  className="sr-only"
                  onChange={handleFallbackFile}
                />
              </label>

              {/* Shutter Button */}
              <button
                type="button"
                onClick={takeSnapshot}
                disabled={!streaming}
                title="Capture Still Frame"
                className="relative flex items-center justify-center w-14 h-14 sm:w-16 sm:h-16 rounded-full border-4 border-white bg-[#0e5a4c] hover:bg-[#127260] active:scale-95 transition disabled:opacity-40 cursor-pointer shadow-md"
              >
                <div className="w-10 h-10 sm:w-12 sm:h-12 rounded-full border-2 border-white/50 flex items-center justify-center">
                  <Camera aria-hidden="true" size={20} className="text-white" />
                </div>
              </button>

              {/* Toggle simulated/real if applicable */}
              <button
                type="button"
                onClick={() => (isSimulated ? void startCamera() : startSimulatedFeed())}
                className="flex items-center gap-1.5 text-xs text-white/70 hover:text-white transition cursor-pointer"
              >
                <Video aria-hidden="true" size={14} />
                <span>{isSimulated ? "Try Camera" : "Demo Simulation"}</span>
              </button>
            </div>
          )}
        </div>
      </div>
      <p id="camera-dialog-description" className="sr-only">Capture a physical camera image, choose an existing file, or explicitly use a generated demonstration frame.</p>
    </div>
  );
}
