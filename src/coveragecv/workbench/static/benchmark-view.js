// Curate one matched comparison per dataset; never mix recipes or unmatched seeds.
export const benchmarkTasks = {
  pawns: {name: 'Chess pawns', detail: '2 classes', methods: ['naive_augmented_512', 'aware_augmented_512', 'complete_augmented_512'], recipe: 'RF-DETR Nano · 512px'},
  'all-pieces': {name: 'All chess pieces', detail: '13 classes', methods: ['naive_large_704_ema', 'aware_large_704_ema', 'complete_reference_large_704_ema'], recipe: 'RF-DETR Large · 704px'},
  construction: {name: 'Construction safety', detail: '5 classes', methods: ['naive', 'aware', 'complete_reference'], recipe: 'RF-DETR Nano · 512px'},
};

export const modelRoles = {
  naive: {label: 'Ordinary training', tag: 'CONTROL', description: 'Standard RF-DETR trained on the incomplete labels.'},
  aware: {label: 'CoverageCV', tag: 'OUR METHOD', description: 'The same labels, with our coverage-aware training.'},
  complete_reference: {label: 'Fully labeled reference', tag: 'REFERENCE', description: 'The same model and images, with all available labels.'},
};

export function matchedBenchmark(data, key) {
  const config = benchmarkTasks[key], task = data.tasks?.[key];
  if (!config || !task) return null;
  const groups = config.methods.map(method => (task.runs ?? []).filter(run => run.method === method));
  const seeds = [...new Set(groups[0].map(run => run.seed))].filter(seed => groups.every(group => group.filter(run => run.seed === seed).length === 1)).sort();
  if (!seeds.length) return null;
  for (const seed of seeds) {
    const rows = groups.map(group => group.find(row => row.seed === seed));
    for (const field of ['initial_parameter_digest', 'steps', 'batch', 'resolution', 'device']) {
      if (rows.some(row => row[field] !== rows[0][field])) throw new Error('Benchmark comparison has mismatched training conditions.');
    }
  }
  const roles = ['naive', 'aware', 'complete_reference'];
  const rows = groups.map((group, index) => {
    const selected = group.filter(run => seeds.includes(run.seed));
    const metrics = {};
    for (const name of ['AP', 'AP50', 'recall_at_threshold']) {
      const values = selected.map(run => run.metrics[name]);
      if (!values.every(value => Number.isFinite(value) && value >= 0 && value <= 1)) throw new Error('Benchmark contains an invalid metric.');
      metrics[name] = values.reduce((sum, value) => sum + value, 0) / values.length;
    }
    return {role: roles[index], ...modelRoles[roles[index]], metrics, runs: selected};
  });
  return {key, ...config, rows, seeds, task, images: rows[0].runs[0].metrics.images, steps: rows[0].runs[0].steps,
    gain: 100 * (rows[1].metrics.AP - rows[0].metrics.AP), gap: 100 * (rows[2].metrics.AP - rows[1].metrics.AP)};
}
