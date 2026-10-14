type MetricsViewProps = {
  metrics: Record<string, unknown> | null;
};

type TargetMetrics = {
  display?: string;
  selected_model?: string;
  cross_validation?: Record<string, number>;
};

export function MetricsView({ metrics }: MetricsViewProps) {
  if (!metrics) return null;

  const keys = ["Cath", "LAD", "LCX", "RCA"];

  return (
    <div className="panel metrics-panel">
      <h2>Model validation (repeated stratified CV)</h2>
      <p className="muted">
        n=303 patients · LAD/LCX/RCA/Cath excluded from inputs · metrics are means ± std
        across folds.
      </p>
      <div className="metrics-grid">
        {keys.map((key) => {
          const block = metrics[key] as TargetMetrics | undefined;
          if (!block?.cross_validation) return null;
          const cv = block.cross_validation;
          return (
            <article key={key} className="metric-card">
              <h3>{block.display ?? key}</h3>
              <p className="model-name">Model: {block.selected_model}</p>
              <ul>
                <li>ROC-AUC: {cv.roc_auc_mean?.toFixed(3)} ± {cv.roc_auc_std?.toFixed(3)}</li>
                <li>F1: {cv.f1_mean?.toFixed(3)} ± {cv.f1_std?.toFixed(3)}</li>
                <li>Recall: {cv.recall_mean?.toFixed(3)} ± {cv.recall_std?.toFixed(3)}</li>
                <li>Precision: {cv.precision_mean?.toFixed(3)} ± {cv.precision_std?.toFixed(3)}</li>
              </ul>
            </article>
          );
        })}
      </div>
    </div>
  );
}
