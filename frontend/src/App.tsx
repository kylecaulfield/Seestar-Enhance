import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { ReactCompareSlider, ReactCompareSliderImage } from "react-compare-slider";

// Same-origin in prod (FastAPI serves the SPA); vite proxy in dev.
const API_BASE = import.meta.env.VITE_API_BASE ?? "";

type View = "drop" | "processing" | "done" | "error";

type Status = {
  job_id: string;
  status: "queued" | "running" | "done" | "error";
  stage: string;
  progress: number;
  classification: string | null;
  error: string | null;
  stages_done: string[];
  // Visible queue (added after the original /status). Optional in
  // the type so older builds of the SPA don't crash if the backend
  // hasn't been redeployed yet.
  queue_position?: number;
  queue_total?: number;
  eta_seconds?: number | null;
  worker_capacity?: number;
};

// Coarse load summary from /health. The drop view fetches this once
// on mount so the UI can warn the user when the pipeline is busy
// before they upload, instead of after.
type Health = {
  status: string;
  load: "idle" | "busy" | "backed_up";
  inflight: number;
  running: number;
  queued: number;
  worker_capacity: number;
  recent_avg_seconds: number;
};

function formatEta(seconds: number | null | undefined): string {
  if (seconds == null || !isFinite(seconds) || seconds <= 0) return "";
  if (seconds < 60) return `~${Math.ceil(seconds)} sec`;
  const mins = Math.ceil(seconds / 60);
  if (mins < 60) return `~${mins} min`;
  const hrs = Math.floor(mins / 60);
  const rem = mins % 60;
  return rem ? `~${hrs} hr ${rem} min` : `~${hrs} hr`;
}

// LCARS-flavoured stage labels. Brief enough to fit the bar.
const STAGE_LABELS: Record<string, string> = {
  queued: "Standing By",
  load: "Subspace Buffer Load",
  classify: "Spectral Class Survey",
  background: "Sky Gradient Compensator",
  color: "Chrominance Calibration",
  stretch: "Tonal Decompression",
  bm3d_denoise: "Noise Filter Bank",
  sharpen: "Edge Acuity Pass",
  curves: "Tonal Curve Engaged",
  export: "Output Buffer Write",
  done: "Operation Complete",
  cosmetic: "Hot-Pixel Sweep",
  dark_subtract: "Dark Frame Subtract",
  spcc: "Photometric Calibration",
  deconv: "PSF Deconvolution",
  stars_split: "Stellar / Diffuse Split",
  starless_stretch: "Diffuse Lift",
  clahe: "Local Contrast",
  recombine: "Stellar Recombine",
};

function stageLabel(stage: string): string {
  return STAGE_LABELS[stage] ?? stage;
}

// Star Trek-style stardate. Loosely TNG era — January 1, 2300 ≈ stardate
// 50000.0 in the on-screen system, with each year ≈ 1000 stardate units.
// Fractional digits = the day-of-year. We pin "now" to JS Date so the
// number ticks while the user has the page open.
function useStardate(): string {
  const [d, setD] = useState(() => new Date());
  useEffect(() => {
    const id = setInterval(() => setD(new Date()), 60000);
    return () => clearInterval(id);
  }, []);
  return useMemo(() => {
    const year = d.getUTCFullYear();
    const startOfYear = Date.UTC(year, 0, 1);
    const dayOfYear = (d.getTime() - startOfYear) / (1000 * 60 * 60 * 24);
    const integer = (year - 2300) * 1000;
    const fractional = (dayOfYear / 365.25) * 1000;
    return (integer + fractional).toFixed(1);
  }, [d]);
}

function LcarsFrame({ section }: { section: string }) {
  const stardate = useStardate();
  return (
    <div className="lcars-frame" aria-hidden>
      <div className="lcars-elbow" />
      <div className="lcars-bar">
        <span>USS Seestar · {section}</span>
        <span className="lcars-bar-stardate">Stardate {stardate}</span>
      </div>
      <div className="lcars-cap" />
    </div>
  );
}

type StageStripProps = {
  jobId: string;
  stages: string[];
};

function StageStrip({ jobId, stages }: StageStripProps) {
  const onImgError = (e: React.SyntheticEvent<HTMLImageElement>) => {
    e.currentTarget.style.visibility = "hidden";
  };
  const onImgLoad = (e: React.SyntheticEvent<HTMLImageElement>) => {
    e.currentTarget.style.visibility = "visible";
  };

  if (stages.length === 0) return null;
  return (
    <div className="stage-strip" aria-label="Pipeline telemetry">
      {stages.map((stage) => (
        <figure key={stage} className="stage-thumb">
          <img
            src={`${API_BASE}/preview/${jobId}/stage/${stage}`}
            alt={stageLabel(stage)}
            loading="lazy"
            onError={onImgError}
            onLoad={onImgLoad}
          />
          <figcaption>{stageLabel(stage)}</figcaption>
        </figure>
      ))}
    </div>
  );
}

type OutputFormat = "png" | "tiff" | "fits";

const OUTPUT_FORMAT_LABELS: Record<OutputFormat, string> = {
  png: "PNG · 16-bit",
  tiff: "TIFF · 16-bit",
  fits: "FITS · float32",
};

export default function App() {
  const [view, setView] = useState<View>("drop");
  const [status, setStatus] = useState<Status | null>(null);
  const [errorMsg, setErrorMsg] = useState<string>("");
  const [dragActive, setDragActive] = useState(false);
  const [resultUrl, setResultUrl] = useState<string>("");
  const [beforeUrl, setBeforeUrl] = useState<string>("");
  const [showStages, setShowStages] = useState<boolean>(false);
  const [health, setHealth] = useState<Health | null>(null);
  // The output format the user picks on the drop view, then carried
  // through processing → done so the Beam Down link references the
  // matching extension. PNG is the default — universal compatibility,
  // smallest size, fine for screen viewing.
  const [outputFormat, setOutputFormat] = useState<OutputFormat>("png");
  const fileInputRef = useRef<HTMLInputElement>(null);
  const pollTimerRef = useRef<ReturnType<typeof setInterval> | null>(null);

  // Refresh the load indicator when the user lands on (or returns to)
  // the drop view. One fetch per visit is enough — the backend updates
  // are bursty (a job per 30-60s), and we don't want a constant poll on
  // an idle page.
  useEffect(() => {
    if (view !== "drop") return;
    let cancelled = false;
    fetch(`${API_BASE}/health`)
      .then((r) => (r.ok ? r.json() : null))
      .then((j: Health | null) => {
        if (!cancelled && j && j.status === "ok") setHealth(j);
      })
      .catch(() => {
        // Health failures are non-fatal — drop view still works without it.
      });
    return () => {
      cancelled = true;
    };
  }, [view]);

  const reset = useCallback(() => {
    if (pollTimerRef.current) {
      clearInterval(pollTimerRef.current);
      pollTimerRef.current = null;
    }
    setStatus(null);
    setErrorMsg("");
    setResultUrl("");
    setBeforeUrl("");
    setShowStages(false);
    setView("drop");
  }, []);

  useEffect(() => {
    return () => {
      if (pollTimerRef.current) clearInterval(pollTimerRef.current);
    };
  }, []);

  const uploadFile = useCallback(async (file: File, format: OutputFormat) => {
    setView("processing");
    setStatus({
      job_id: "",
      status: "queued",
      stage: "queued",
      progress: 0,
      classification: null,
      error: null,
      stages_done: [],
    });

    try {
      const form = new FormData();
      form.append("file", file);
      const resp = await fetch(
        `${API_BASE}/process?format=${encodeURIComponent(format)}`,
        { method: "POST", body: form },
      );
      if (!resp.ok) {
        const text = await resp.text().catch(() => "");
        throw new Error(text || `upload failed: ${resp.status}`);
      }
      const { job_id } = (await resp.json()) as { job_id: string };

      pollTimerRef.current = setInterval(async () => {
        try {
          const r = await fetch(`${API_BASE}/status/${job_id}`);
          if (!r.ok) throw new Error(`status ${r.status}`);
          const s = (await r.json()) as Status;
          setStatus(s);
          if (s.status === "done") {
            if (pollTimerRef.current) {
              clearInterval(pollTimerRef.current);
              pollTimerRef.current = null;
            }
            setResultUrl(`${API_BASE}/result/${job_id}`);
            setBeforeUrl(`${API_BASE}/preview/${job_id}/before`);
            setView("done");
          } else if (s.status === "error") {
            if (pollTimerRef.current) {
              clearInterval(pollTimerRef.current);
              pollTimerRef.current = null;
            }
            setErrorMsg(s.error ?? "pipeline failed");
            setView("error");
          }
        } catch (err) {
          // transient; keep polling
          console.warn("poll error", err);
        }
      }, 500);
    } catch (err) {
      setErrorMsg(err instanceof Error ? err.message : "upload failed");
      setView("error");
    }
  }, []);

  const onPick = useCallback(
    (file: File | undefined) => {
      if (!file) return;
      const name = file.name.toLowerCase();
      if (!(name.endsWith(".fit") || name.endsWith(".fits") || name.endsWith(".fts"))) {
        setErrorMsg("Subspace transfer rejected: expected .fit, .fits, or .fts");
        setView("error");
        return;
      }
      void uploadFile(file, outputFormat);
    },
    [uploadFile, outputFormat],
  );

  const onDrop = useCallback(
    (e: React.DragEvent<HTMLDivElement>) => {
      e.preventDefault();
      setDragActive(false);
      onPick(e.dataTransfer.files?.[0]);
    },
    [onPick],
  );

  const onDragOver = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setDragActive(true);
  };
  const onDragLeave = () => setDragActive(false);

  if (view === "drop") {
    return (
      <>
        <LcarsFrame section="Awaiting Image File" />
        <main className="page">
          <div
            className={`drop-zone${dragActive ? " drop-zone--active" : ""}`}
            onDrop={onDrop}
            onDragOver={onDragOver}
            onDragLeave={onDragLeave}
            onClick={() => fileInputRef.current?.click()}
            role="button"
            tabIndex={0}
          >
            <div className="drop-icon" aria-hidden>
              ⬈
            </div>
            <div className="drop-title">Awaiting Image File</div>
            <div className="drop-sub">Deposit FITS · or tap to browse</div>
            <input
              ref={fileInputRef}
              type="file"
              accept=".fit,.fits,.fts"
              className="hidden-input"
              onChange={(e) => onPick(e.target.files?.[0])}
            />
          </div>

          <div className="format-row" onClick={(e) => e.stopPropagation()}>
            <label className="format-label" htmlFor="format-select">
              Output Format
            </label>
            <select
              id="format-select"
              className="format-select"
              value={outputFormat}
              onChange={(e) => setOutputFormat(e.target.value as OutputFormat)}
            >
              {(Object.keys(OUTPUT_FORMAT_LABELS) as OutputFormat[]).map((f) => (
                <option key={f} value={f}>
                  {OUTPUT_FORMAT_LABELS[f]}
                </option>
              ))}
            </select>
          </div>
          {health && health.load !== "idle" && (
            <div className={`load-badge load-${health.load}`}>
              {health.load === "busy"
                ? `Pipeline busy · ${health.running} job${health.running === 1 ? "" : "s"} running`
                : (() => {
                    // Wait estimate when joining the back of the queue:
                    // ceil(queued / workers) waves of avg time. Without
                    // the divide-by-workers we'd overstate wait by the
                    // worker count (e.g. 6 queued / 2 workers = 3 waves,
                    // not 6).
                    const workers = Math.max(1, health.worker_capacity);
                    const waves = Math.ceil(health.queued / workers);
                    const wait = waves * health.recent_avg_seconds;
                    return `Pipeline backed up · ${health.inflight} jobs in flight · ${formatEta(wait)} wait`;
                  })()}
            </div>
          )}
          <div className="brand">USS Seestar · Image Enhancement Subsystem</div>
        </main>
      </>
    );
  }

  if (view === "processing") {
    const pct = Math.round((status?.progress ?? 0) * 100);
    const jobId = status?.job_id ?? "";
    const stagesDone = status?.stages_done ?? [];
    const queuePos = status?.queue_position ?? 0;
    const queueTotal = status?.queue_total ?? 0;
    const workerCap = status?.worker_capacity ?? 1;
    const eta = formatEta(status?.eta_seconds);
    // We're "actually queued" (not running) when our position is past
    // the worker capacity. Position 1..workerCap means a worker is
    // already crunching us — show the stage label, not the queue line.
    const isQueuedBehind = queuePos > workerCap;
    return (
      <>
        <LcarsFrame section={isQueuedBehind ? "Standing By" : "Pipeline Engaged"} />
        <main className="page page--center">
          <div className="processing">
            <div className="spinner" aria-hidden />
            {isQueuedBehind ? (
              <>
                <div className="processing-title">
                  Position {queuePos} of {queueTotal}
                </div>
                {eta && (
                  <div className="processing-meta">Estimated wait · {eta}</div>
                )}
              </>
            ) : (
              <>
                <div className="processing-title">
                  {stageLabel(status?.stage ?? "queued")}
                </div>
                <div className="processing-bar" aria-hidden>
                  <div className="processing-bar-fill" style={{ width: `${pct}%` }} />
                </div>
                {eta && (
                  <div className="processing-meta">Remaining · {eta}</div>
                )}
              </>
            )}
            {status?.classification && (
              <div className="processing-meta">
                Profile · <span className="accent">{status.classification}</span>
              </div>
            )}
            {jobId && stagesDone.length > 0 && (
              <StageStrip jobId={jobId} stages={stagesDone} />
            )}
          </div>
        </main>
      </>
    );
  }

  if (view === "done") {
    const jobId = status?.job_id ?? "";
    const stagesDone = status?.stages_done ?? [];
    return (
      <>
        <LcarsFrame section="Image Enhancement Complete" />
        <main className="page page--done">
          <div className="result-wrap">
            <ReactCompareSlider
              itemOne={
                <ReactCompareSliderImage
                  src={beforeUrl}
                  alt="Original signal"
                  style={{ backgroundColor: "#000" }}
                />
              }
              itemTwo={
                <ReactCompareSliderImage
                  src={resultUrl}
                  alt="Enhanced output"
                  style={{ backgroundColor: "#000" }}
                />
              }
              style={{ height: "100%", width: "100%" }}
            />
          </div>
          {showStages && jobId && stagesDone.length > 0 && (
            <div className="stage-strip-overlay">
              <StageStrip jobId={jobId} stages={stagesDone} />
            </div>
          )}
          <div className="toolbar">
            <a className="btn btn-primary" href={resultUrl} download>
              Beam Down
            </a>
            {stagesDone.length > 0 && (
              <button
                className="btn btn-ghost btn-tertiary"
                onClick={() => setShowStages((v) => !v)}
              >
                {showStages ? "Hide Telemetry" : "Show Telemetry"}
              </button>
            )}
            <button className="btn btn-ghost" onClick={reset}>
              New Subspace Capture
            </button>
          </div>
        </main>
      </>
    );
  }

  return (
    <>
      <LcarsFrame section="Anomaly Detected" />
      <main className="page page--center">
        <div className="error-card">
          <div className="error-title">Subsystem Fault</div>
          <div className="error-msg">{errorMsg}</div>
          <button className="btn btn-primary" onClick={reset}>
            Re-engage
          </button>
        </div>
      </main>
    </>
  );
}
