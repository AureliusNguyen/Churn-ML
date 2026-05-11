export function Colophon() {
  return (
    <footer className="pt-12">
      <div className="grid grid-cols-1 gap-6 border-t-2 border-ink pt-6 text-[12px] leading-[1.6] text-graph sm:grid-cols-[1.5fr_1fr_1fr]">
        <div>
          <div className="font-display text-base text-ink">The Churn Report</div>
          <p className="mt-1 max-w-[44ch] italic">
            An editorial dashboard for predicting and explaining bank customer
            churn. Inputs flow to a Python service that runs ten scikit-learn
            and XGBoost models. Charts and animations are rendered locally.
          </p>
        </div>
        <div>
          <div className="text-[11px] uppercase tracking-[0.18em] text-mute">
            Models in this issue
          </div>
          <ul className="mt-2 space-y-1 font-mono text-[11px] tabular text-graph">
            <li>XGBoost &middot; XGBoost (FE) &middot; XGBoost (SMOTE)</li>
            <li>Best XGB &middot; Stacking</li>
            <li>Random Forest &middot; KNN &middot; SVM</li>
            <li>Voting classifier (advisory)</li>
          </ul>
        </div>
        <div>
          <div className="text-[11px] uppercase tracking-[0.18em] text-mute">Data</div>
          <p className="mt-2 font-mono text-[11px] tabular text-graph">
            churn.csv &middot; 10,000 rows
            <br />
            target: Exited
          </p>
          <div className="mt-3 text-[11px] uppercase tracking-[0.18em] text-mute">
            Stack
          </div>
          <p className="mt-2 font-mono text-[11px] tabular text-graph">
            Next.js &middot; Tailwind &middot; Motion &middot; FastAPI
          </p>
        </div>
      </div>
    </footer>
  );
}
