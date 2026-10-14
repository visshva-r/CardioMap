import { useMemo } from "react";
import type { SchemaField } from "../api";

const GROUP_RULES: { title: string; names: string[] }[] = [
  {
    title: "Demographics",
    names: ["Age", "Weight", "Length", "Sex", "BMI"],
  },
  {
    title: "Symptoms & examination",
    names: [
      "DM",
      "HTN",
      "Current Smoker",
      "EX-Smoker",
      "FH",
      "Obesity",
      "CRF",
      "CVA",
      "Airway disease",
      "Thyroid Disease",
      "CHF",
      "DLP",
      "BP",
      "PR",
      "Edema",
      "Weak Peripheral Pulse",
      "Lung rales",
      "Systolic Murmur",
      "Diastolic Murmur",
      "Typical Chest Pain",
      "Dyspnea",
      "Function Class",
      "Atypical",
      "Nonanginal",
      "Exertional CP",
      "LowTH Ang",
    ],
  },
  {
    title: "ECG",
    names: [
      "Q Wave",
      "St Elevation",
      "St Depression",
      "Tinversion",
      "LVH",
      "Poor R Progression",
      "BBB",
    ],
  },
  {
    title: "Laboratory & echo",
    names: [
      "FBS",
      "CR",
      "TG",
      "LDL",
      "HDL",
      "BUN",
      "ESR",
      "HB",
      "K",
      "Na",
      "WBC",
      "Lymph",
      "Neut",
      "PLT",
      "EF-TTE",
      "Region RWMA",
      "VHD",
    ],
  },
];

function groupFields(schema: SchemaField[]) {
  const map = new Map<string, SchemaField[]>();
  for (const rule of GROUP_RULES) {
    map.set(
      rule.title,
      schema.filter((f) => rule.names.includes(f.name))
    );
  }
  const assigned = new Set(GROUP_RULES.flatMap((g) => g.names));
  const rest = schema.filter((f) => !assigned.has(f.name));
  if (rest.length) map.set("Other", rest);
  return map;
}

type PatientFormProps = {
  schema: SchemaField[];
  values: Record<string, string | number>;
  onChange: (name: string, value: string | number) => void;
  onPredict: () => void;
  loading: boolean;
  samples: { index: number; features: Record<string, string | number> }[];
  selectedSampleIndex: number | null;
  onLoadSample: (index: number, features: Record<string, string | number>) => void;
};

export function PatientForm({
  schema,
  values,
  onChange,
  onPredict,
  loading,
  samples,
  selectedSampleIndex,
  onLoadSample,
}: PatientFormProps) {
  const groups = useMemo(() => groupFields(schema), [schema]);

  return (
    <div className="panel patient-form">
      <div className="panel-header">
        <h2>Patient profile</h2>
        <select
          className="sample-select"
          value={selectedSampleIndex ?? ""}
          onChange={(e) => {
            const idx = Number(e.target.value);
            const sample = samples.find((s) => s.index === idx);
            if (sample) onLoadSample(sample.index, sample.features);
          }}
        >
          <option value="" disabled>
            Load sample patient…
          </option>
          {samples.map((s) => (
            <option key={s.index} value={s.index}>
              Record #{s.index + 1}
            </option>
          ))}
        </select>
      </div>
      <div className="form-scroll">
        {Array.from(groups.entries()).map(([title, fields]) => (
          <details key={title} open={title === "Demographics"}>
            <summary>{title}</summary>
            <div className="field-grid">
              {fields.map((field) => (
                <label key={field.name} className="field">
                  <span>{field.name}</span>
                  {field.type === "number" ? (
                    <input
                      type="number"
                      step="any"
                      min={field.min}
                      max={field.max}
                      value={values[field.name] ?? field.default ?? 0}
                      onChange={(e) => onChange(field.name, Number(e.target.value))}
                    />
                  ) : (
                    <select
                      value={String(values[field.name] ?? field.default ?? "")}
                      onChange={(e) => onChange(field.name, e.target.value)}
                    >
                      {(field.options ?? []).map((opt) => (
                        <option key={opt} value={opt}>
                          {opt}
                        </option>
                      ))}
                    </select>
                  )}
                </label>
              ))}
            </div>
          </details>
        ))}
      </div>
      <button type="button" className="primary-btn" onClick={onPredict} disabled={loading}>
        {loading ? "Running models…" : "Predict risk"}
      </button>
    </div>
  );
}
