// Runs the browser evaluator over every committed bundle and compares its output to
// the metrics the Python scorer published. Exits non-zero unless every run matches,
// so "the site recomputes the real number" stays a tested claim and not a hope.
import {readFileSync} from 'node:fs';
import {fileURLToPath} from 'node:url';
import {dirname, join} from 'node:path';
import {decodePredictions, evaluate} from '../src/coveragecv/workbench/static/coco-eval.js';

const ROOT = join(dirname(fileURLToPath(import.meta.url)), '..');
const DIR = process.env.VERIFY_DIR ?? join(ROOT, 'src/coveragecv/workbench/static/verify');
const manifest = JSON.parse(readFileSync(join(DIR, 'manifest.json'), 'utf8'));
const TOLERANCE = Number(process.env.PARITY_TOLERANCE ?? 0);

const labels = new Map();
for (const [digest, entry] of Object.entries(manifest.ground_truth)) {
  labels.set(digest, JSON.parse(readFileSync(join(DIR, entry.path), 'utf8')));
}

const COMPARED = ['AP', 'AP50', 'AP75', 'AR100', 'precision_at_threshold', 'recall_at_threshold',
                  'false_positives_per_image', 'negative_images',
                  'negative_image_false_positives_per_image', 'images', 'boxes'];

let checked = 0, failed = 0, worst = 0, worstLabel = '';
for (const cohort of manifest.cohorts) {
  for (const run of cohort.runs) {
    const bytes = readFileSync(join(DIR, run.predictions.path));
    const buffer = bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength);
    const predictions = decodePredictions(buffer);
    if (predictions.length !== run.predictions.count) {
      console.error(`FAIL ${run.id}: decoded ${predictions.length}, manifest says ${run.predictions.count}`);
      failed += 1;
      continue;
    }
    const actual = evaluate(predictions, labels.get(run.bundle_digest), {threshold: run.score_threshold});
    const problems = [];
    for (const key of COMPARED) {
      const diff = Math.abs(actual[key] - run.expected[key]);
      if (diff > worst) { worst = diff; worstLabel = `${run.id}.${key}`; }
      if (!(diff <= TOLERANCE)) problems.push(`${key}: expected ${run.expected[key]}, got ${actual[key]}`);
    }
    for (const [name, expected] of Object.entries(run.expected.per_class_AP)) {
      const got = actual.per_class_AP[name];
      if (expected === null || got === null) {
        if (expected !== got) problems.push(`per_class_AP.${name}: expected ${expected}, got ${got}`);
        continue;
      }
      const diff = Math.abs(got - expected);
      if (diff > worst) { worst = diff; worstLabel = `${run.id}.per_class_AP.${name}`; }
      if (!(diff <= TOLERANCE)) problems.push(`per_class_AP.${name}: expected ${expected}, got ${got}`);
    }
    for (const [key, expected] of Object.entries(run.expected.class_support)) {
      if (actual.class_support[key] !== expected) problems.push(`class_support.${key}`);
    }
    checked += 1;
    if (problems.length) {
      failed += 1;
      console.error(`FAIL ${run.id}\n  ${problems.join('\n  ')}`);
    } else {
      console.log(`ok   ${run.id.padEnd(44)} AP ${(100 * actual.AP).toFixed(4)}`);
    }
  }
}
console.log(`\n${checked - failed}/${checked} runs reproduced exactly (tolerance ${TOLERANCE}).`);
console.log(`Largest deviation across every compared field: ${worst} (${worstLabel || 'none'})`);
if (!checked) { console.error('No run was checked; refusing to report success.'); process.exit(1); }
if (failed) process.exit(1);
