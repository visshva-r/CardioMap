export type SchemaField = {
  name: string;
  type: "number" | "category";
  min?: number;
  max?: number;
  default?: number | string;
  options?: string[];
};

export type SchemaResponse = {
  schema: SchemaField[];
  targets: Record<string, { display: string }>;
};

export type PredictionTarget = {
  label: string;
  probability: number;
  risk_level: string;
  contributions: {
    feature: string;
    shap: number;
    direction: string;
  }[];
};

export type PredictResponse = {
  predictions: Record<string, PredictionTarget>;
};

export async function fetchSchema(): Promise<SchemaResponse> {
  const res = await fetch("/api/schema");
  if (!res.ok) throw new Error("Schema unavailable. Train models first.");
  return res.json();
}

export async function fetchMetrics(): Promise<Record<string, unknown>> {
  const res = await fetch("/api/metrics");
  if (!res.ok) throw new Error("Metrics unavailable.");
  return res.json();
}

export async function fetchSamples(): Promise<
  { index: number; features: Record<string, string | number> }[]
> {
  const res = await fetch("/api/patients/samples");
  if (!res.ok) throw new Error("Sample patients unavailable.");
  return res.json();
}

export async function predict(
  features: Record<string, string | number>
): Promise<PredictResponse> {
  const res = await fetch("/api/predict", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ features }),
  });
  if (!res.ok) {
    const text = await res.text();
    throw new Error(text || "Prediction failed");
  }
  return res.json();
}
