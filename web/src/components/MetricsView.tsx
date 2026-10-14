type MetricsViewProps = {
  metrics: Record<string, unknown> | null;
};

type TargetMetrics = {
  display?: string;
  selected_model?: string;
  cross_validation?: Record<string, number>;
};

const TARGETS = ["Cath", "LAD", "LCX", "RCA"] as const;

export function MetricsView({ metrics }: MetricsViewProps) {
  if (!metrics) return null;

  return (
    <div className="panel metrics-panel">
      <h2>Model validation (repeated stratified CV)</h2>
      <p className="muted">
        n=303 patients · LAD/LCX/RCA/Cath excluded from inputs · table shows mean ± std
        from 5×3 repeated CV. Plots show single 5-fold out-of-fold curves for the selected
        model per target.
      </p>
      <div className="metrics-grid">
        {TARGETS.map((key) => {
          const block = metrics[key] as TargetMetrics | undefined;
          if (!block?.cross_validation) return null;
          const cv = block.cross_validation;
          const slug = key.toLowerCase();
          return (
            <article key={key} className="metric-card metric-card-wide">
              <h3>{block.display ?? key}</h3>
              <p className="model-name">Model: {block.selected_model}</p>
              <ul>
                <li>
                  ROC-AUC: {cv.roc_auc_mean?.toFixed(3)} ± {cv.roc_auc_std?.toFixed(3)}
                </li>
                <li>
                  F1: {cv.f1_mean?.toFixed(3)} ± {cv.f1_std?.toFixed(3)}
                </li>
                <li>
                  Recall: {cv.recall_mean?.toFixed(3)} ± {cv.recall_std?.toFixed(3)}
                </li>
                <li>
                  Precision: {cv.precision_mean?.toFixed(3)} ± {cv.precision_std?.toFixed(3)}
                </li>
              </ul>
              <div className="metric-figures">
                <figure>
                  <img src={`/figures/${slug}_roc.png`} alt={`${key} OOF ROC curve`} />
                  <figcaption>OOF ROC (5-fold)</figcaption>
                </figure>
                <figure>
                  <img src={`/figures/${slug}_cm.png`} alt={`${key} confusion matrix`} />
                  <figcaption>OOF confusion @ 0.5</figcaption>
                </figure>
              </div>
            </article>
          );
        })}
      </div>
    </div>
  );
}
