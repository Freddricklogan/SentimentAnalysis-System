/** Report page behaviour: mounts the Executive Shell from the embedded JSON. */
import { mountExecShell } from './exec-shell.js';

const data = JSON.parse(document.getElementById('report-data').textContent);
const pct = (v) => `${(v * 100).toFixed(2)}%`;

const shell = mountExecShell({
  title: 'Sentiment Analysis System',
  tagline: 'A fine-tuned DistilBERT classifier behind a FastAPI service with Pydantic v2 models and a batch endpoint, evaluated in CI against the SST-2 development set beside a VADER lexicon baseline. The API runs locally or in Docker; this page is the evaluation it publishes.',
  repo: 'https://github.com/Freddricklogan/SentimentAnalysis-System',
  pagesUrl: 'https://freddricklogan.github.io/SentimentAnalysis-System/',
  badges: [{ label: 'DistilBERT · SST-2', tone: 'accent' }, { label: 'FastAPI + Pydantic v2', dot: true }, { label: 'Measured in CI', dot: true }],
  kpis: [
    { label: 'DistilBERT F1', compute: () => data.transformer.f1.toFixed(4), tone: 'accent' },
    { label: 'DistilBERT accuracy', compute: () => pct(data.transformer.accuracy), tone: 'ok' },
    { label: 'VADER F1', compute: () => data.baseline.f1.toFixed(4) },
    { label: 'Majority class', compute: () => pct(data.majority), tone: 'muted' },
    { label: 'Sentences / s (CPU)', compute: () => Math.round(data.transformer.per_second), tone: 'warn' }
  ],
  tour: [
    { selector: '.ss-note', title: 'What this page is', body: `The CI run evaluated ${data.n} labelled sentences with the pinned model ${data.model}. The API is not hosted here; the report is what it publishes.` },
    { selector: '#s-scores', title: 'Against a baseline that costs nothing', body: `VADER, a lexicon with no training, reaches F1 ${data.baseline.f1.toFixed(3)}; the transformer reaches ${data.transformer.f1.toFixed(3)}. The table shows the confusion counts and throughput behind both.` },
    { selector: '#s-conflicts', title: 'Where they disagree', body: 'Sentences the two classifiers label differently, with the truth — the cases where a lexicon\'s word counting fails and a fine-tuned model does not, or vice versa.' }
  ]
});
shell.refreshKpis();
