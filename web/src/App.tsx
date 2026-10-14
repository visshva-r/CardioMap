import { useCallback, useEffect, useMemo, useState } from "react";
import {
  fetchMetrics,
  fetchSamples,
  fetchSchema,
  predict,
  type PredictResponse,
  type SchemaField,
} from "./api";
import type { VesselId } from "./anatomy/registry";
import { Disclaimer } from "./components/Disclaimer";
import { HeartScene } from "./components/HeartScene";
import { MetricsView } from "./components/MetricsView";
import { PatientForm } from "./components/PatientForm";
import { RiskPanel } from "./components/RiskPanel";
import { ShapChart } from "./components/ShapChart";
import "./App.css";

function defaultsFromSchema(schema: SchemaField[]) {
  const out: Record<string, string | number> = {};
  for (const f of schema) {
    out[f.name] = f.default ?? (f.type === "number" ? 0 : "");
  }
  return out;
}

export default function App() {
  const [schema, setSchema] = useState<SchemaField[]>([]);
  const [values, setValues] = useState<Record<string, string | number>>({});
  const [samples, setSamples] = useState<
    { index: number; features: Record<string, string | number> }[]
  >([]);
  const [result, setResult] = useState<PredictResponse | null>(null);
  const [metrics, setMetrics] = useState<Record<string, unknown> | null>(null);
  const [selected, setSelected] = useState<VesselId | "Cath">("Cath");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [tab, setTab] = useState<"dashboard" | "metrics">("dashboard");
  const [selectedSampleIndex, setSelectedSampleIndex] = useState<number | null>(null);

  useEffect(() => {
    (async () => {
      try {
        const [schemaRes, sampleList, metricJson] = await Promise.all([
          fetchSchema(),
          fetchSamples(),
          fetchMetrics(),
        ]);
        setSchema(schemaRes.schema);
        setValues(defaultsFromSchema(schemaRes.schema));
        setSamples(sampleList);
        setMetrics(metricJson);
      } catch (e) {
        setError(e instanceof Error ? e.message : "Failed to load API");
      }
    })();
  }, []);

  const probabilities = useMemo(() => {
    const base = { LAD: 0.15, LCX: 0.15, RCA: 0.15 };
    if (!result) return base;
    return {
      LAD: result.predictions.LAD.probability,
      LCX: result.predictions.LCX.probability,
      RCA: result.predictions.RCA.probability,
    };
  }, [result]);

  const selectedTarget = useMemo(() => {
    if (!result) return null;
    return result.predictions[selected];
  }, [result, selected]);

  const onPredict = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await predict(values);
      setResult(res);
      setSelected("Cath");
    } catch (e) {
      setError(e instanceof Error ? e.message : "Prediction failed");
    } finally {
      setLoading(false);
    }
  }, [values]);

  return (
    <div className="app">
      <header className="app-header">
        <div>
          <h1>CardioMap</h1>
          <p>Track A · Multimodal AI Hackathon 2026 · CAD & vessel stenosis visualization</p>
        </div>
        <nav className="tabs">
          <button
            type="button"
            className={tab === "dashboard" ? "active" : ""}
            onClick={() => setTab("dashboard")}
          >
            Dashboard
          </button>
          <button
            type="button"
            className={tab === "metrics" ? "active" : ""}
            onClick={() => setTab("metrics")}
          >
            Validation
          </button>
        </nav>
      </header>

      <Disclaimer />

      {error && <div className="error-banner">{error}</div>}

      {tab === "metrics" ? (
        <MetricsView metrics={metrics} />
      ) : (
        <main className="layout">
          <PatientForm
            schema={schema}
            values={values}
            onChange={(name, value) => setValues((v) => ({ ...v, [name]: value }))}
            onPredict={onPredict}
            loading={loading}
            samples={samples}
            selectedSampleIndex={selectedSampleIndex}
            onLoadSample={(index, features) => {
              setSelectedSampleIndex(index);
              setValues(features);
            }}
          />
          <section className="center-column">
            <HeartScene
              probabilities={probabilities}
              selected={selected}
              onSelect={(id) => setSelected(id)}
            />
            <RiskPanel result={result} selected={selected} onSelect={setSelected} />
          </section>
          <div className="panel explain-panel">
            <ShapChart target={selectedTarget} />
          </div>
        </main>
      )}
    </div>
  );
}
