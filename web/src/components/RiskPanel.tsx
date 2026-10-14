import type { PredictResponse } from "../api";
import type { VesselId } from "../anatomy/registry";
import { VESSELS } from "../anatomy/registry";

type RiskPanelProps = {
  result: PredictResponse | null;
  selected: VesselId | "Cath";
  onSelect: (id: VesselId | "Cath") => void;
};

function barClass(level: string) {
  if (level === "high") return "risk-high";
  if (level === "moderate") return "risk-mod";
  return "risk-low";
}

export function RiskPanel({ result, selected, onSelect }: RiskPanelProps) {
  if (!result) {
    return (
      <div className="panel risk-panel">
        <h2>Risk summary</h2>
        <p className="muted">Predictions appear here after you run the models.</p>
      </div>
    );
  }

  const cath = result.predictions.Cath;

  return (
    <div className="panel risk-panel">
      <h2>Risk summary</h2>
      <button
        type="button"
        className={`risk-card overall ${selected === "Cath" ? "selected" : ""}`}
        onClick={() => onSelect("Cath")}
      >
        <span className="risk-label">{cath.label}</span>
        <span className={`risk-value ${barClass(cath.risk_level)}`}>
          {(cath.probability * 100).toFixed(1)}%
        </span>
        <span className="risk-level">{cath.risk_level} risk</span>
      </button>
      <div className="vessel-list">
        {VESSELS.map((v) => {
          const p = result.predictions[v.targetKey];
          return (
            <button
              type="button"
              key={v.id}
              className={`risk-card ${selected === v.id ? "selected" : ""}`}
              onClick={() => onSelect(v.id)}
            >
              <span className="risk-label">{v.label}</span>
              <span className={`risk-value ${barClass(p.risk_level)}`}>
                {(p.probability * 100).toFixed(1)}%
              </span>
              <div className="mini-bar">
                <div className={barClass(p.risk_level)} style={{ width: `${p.probability * 100}%` }} />
              </div>
            </button>
          );
        })}
      </div>
    </div>
  );
}
