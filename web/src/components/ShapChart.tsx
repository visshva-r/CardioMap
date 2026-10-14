import type { PredictionTarget } from "../api";

type ShapChartProps = {
  target: PredictionTarget | null;
};

export function ShapChart({ target }: ShapChartProps) {
  if (!target) {
    return <p className="muted">Run a prediction to see SHAP-style contributions.</p>;
  }

  const maxAbs = Math.max(...target.contributions.map((c) => Math.abs(c.shap)), 0.01);

  return (
    <div className="shap-chart">
      <h3>{target.label} · top drivers</h3>
      <ul>
        {target.contributions.map((c) => (
          <li key={c.feature}>
            <div className="shap-row">
              <span className="shap-name" title={c.feature}>
                {c.feature.replace(/^cat__|^num__/, "")}
              </span>
              <div className="shap-bar-wrap">
                <div
                  className={`shap-bar ${c.shap >= 0 ? "pos" : "neg"}`}
                  style={{ width: `${(Math.abs(c.shap) / maxAbs) * 100}%` }}
                />
              </div>
              <span className="shap-val">{c.shap.toFixed(3)}</span>
            </div>
          </li>
        ))}
      </ul>
    </div>
  );
}
