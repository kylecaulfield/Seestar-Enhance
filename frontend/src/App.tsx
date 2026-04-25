import { useCallback, useEffect, useRef, useState } from "react";
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
};

const STAGE_LABELS: Record<string, string> = {
  queued: "Queued",
  load: "Loading FITS",
  classify: "Classifying image",
  background: "Removing sky gradient",
  color: "Balancing color",
  stretch: "Stretching",
  bm3d_denoise: "Denoising",
  sharpen: "Sharpening",
  curves: "Applying curves",
  export: "Writing PNG",
  done: "Done",
  cosmetic: "Hot-pixel cleanup",
  dark_subtract: "Dark frame",
  spcc: "SPCC calibration",
  deconv: "Deconvolution",
  stars_split: "Star split",
  starless_stretch: "Starless stretch",
  clahe: "CLAHE",
  recombine: "Recombine stars",
};

function stageLabel(stage: string): string {
  return STAGE_LABELS[stage] ?? stage;
}

type StageStripProps = {
  jobId: string;
  stages: string[];
};

function StageStrip({ jobId, stages }: StageStripProps) {
  if (stages.length === 0) return null;
  return (
    <div className="stage-strip" aria-label="Pipeline stages">
      {stages.map((stage) => (
        <figure key={stage} className="stage-thumb">
          <img
            src={`${API_BASE}/preview/${jobId}/stage/${stage}`}
            alt={stageLabel(stage)}
            loading="lazy"
          />
          <figcaption>{stageLabel(stage)}</figcaption>
        </figure>
      ))}
    </div>
  );
}

export default function App() {
  const [view, setView] = useState<View>("drop");
  const [status, setStatus] = useState<Status | null>(null);
  const [errorMsg, setErrorMsg] = useState<string>("");
  const [dragActive, setDragActive] = useState(false);
  const [resultUrl, setResultUrl] = useState<string>("");
  const [beforeUrl, setBeforeUrl] = useState<string>("");
  const [showStages, setShowStages] = useState<boolean>(false);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const pollTimerRef = useRef<ReturnType<typeof setInterval> | null>(null);

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

  const uploadFile = useCallback(async (file: File) => {
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
      const resp = await fetch(`${API_BASE}/process`, {
        method: "POST",
        body: form,
      });
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
        setErrorMsg("Please drop a .fit, .fits, or .fts file.");
        setView("error");
        return;
      }
      void uploadFile(file);
    },
    [uploadFile],
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
          <div className="drop-title">Drop your Seestar FITS file here</div>
          <div className="drop-sub">or click to browse</div>
          <input
            ref={fileInputRef}
            type="file"
            accept=".fit,.fits,.fts"
            className="hidden-input"
            onChange={(e) => onPick(e.target.files?.[0])}
          />
        </div>
        <div className="brand">Seestar Enhance</div>
      </main>
    );
  }

  if (view === "processing") {
    const pct = Math.round((status?.progress ?? 0) * 100);
    const jobId = status?.job_id ?? "";
    const stagesDone = status?.stages_done ?? [];
    return (
      <main className="page page--center">
        <div className="processing">
          <div className="spinner" aria-hidden />
          <div className="processing-title">{stageLabel(status?.stage ?? "queued")}</div>
          <div className="processing-bar" aria-hidden>
            <div className="processing-bar-fill" style={{ width: `${pct}%` }} />
          </div>
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
    );
  }

  if (view === "done") {
    const jobId = status?.job_id ?? "";
    const stagesDone = status?.stages_done ?? [];
    return (
      <main className="page page--done">
        <div className="result-wrap">
          <ReactCompareSlider
            itemOne={
              <ReactCompareSliderImage
                src={beforeUrl}
                alt="Before"
                style={{ backgroundColor: "#000" }}
              />
            }
            itemTwo={
              <ReactCompareSliderImage
                src={resultUrl}
                alt="After"
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
            Download PNG
          </a>
          {stagesDone.length > 0 && (
            <button
              className="btn btn-ghost"
              onClick={() => setShowStages((v) => !v)}
            >
              {showStages ? "Hide stages" : "Show stages"}
            </button>
          )}
          <button className="btn btn-ghost" onClick={reset}>
            Process another
          </button>
        </div>
      </main>
    );
  }

  return (
    <main className="page page--center">
      <div className="error-card">
        <div className="error-title">Something went wrong</div>
        <div className="error-msg">{errorMsg}</div>
        <button className="btn btn-primary" onClick={reset}>
          Try again
        </button>
      </div>
    </main>
  );
}
