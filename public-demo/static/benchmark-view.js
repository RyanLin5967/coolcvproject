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

// Independently selected AP50:95 extrema from the published results ledger.
// Source ledger SHA-256: f34b93d40019b82441e78433acb6346d220aabfe0c1c28eda75c0cac4de352d8
export const recordedExtrema = {
  "all-pieces": {
    "naive": {
      "metrics": {
        "AP": 0.5673147014337226,
        "AP50": 0.7266028851551712,
        "recall_at_threshold": 0.621268656716418
      },
      "method": "naive",
      "seed": 20260919,
      "steps": 2000,
      "resolution": 384,
      "passes": 1,
      "note": "",
      "source": "/tasks/all-pieces/runs/2/metrics",
      "selection": "minimum"
    },
    "aware": {
      "metrics": {
        "AP": 0.741022965108933,
        "AP50": 0.9049523853353046,
        "recall_at_threshold": 0.9888059701492538
      },
      "method": "research_v2_exclusive",
      "seed": 20260918,
      "steps": 4000,
      "resolution": 704,
      "passes": 1,
      "note": "",
      "source": "/research_v2/tasks/all-pieces/exclusive/metrics",
      "selection": "maximum"
    },
    "complete_reference": {
      "metrics": {
        "AP": 0.72475490909525,
        "AP50": 0.8970232603305641,
        "recall_at_threshold": 0.9813432835820896
      },
      "method": "complete_reference",
      "seed": 20260917,
      "steps": 2000,
      "resolution": 384,
      "passes": 1,
      "note": "",
      "source": "/tasks/all-pieces/runs/6/metrics",
      "selection": "minimum"
    }
  },
  "construction": {
    "naive": {
      "metrics": {
        "AP": 0.42311665013241967,
        "AP50": 0.8237463627794711,
        "recall_at_threshold": 0.8131101813110181
      },
      "method": "naive_object_crops",
      "seed": 20260917,
      "steps": 6000,
      "resolution": 512,
      "passes": 5,
      "note": "Fixed tiled inference",
      "source": "/construction_crop_tiled_interaction/runs/2/tiled_metrics",
      "selection": "minimum"
    },
    "aware": {
      "metrics": {
        "AP": 0.5209427679705265,
        "AP50": 0.9254011902222609,
        "recall_at_threshold": 0.9288702928870293
      },
      "method": "research_v2_exclusive",
      "seed": 20260918,
      "steps": 8000,
      "resolution": 512,
      "passes": 5,
      "note": "265 additional published training boxes",
      "source": "/research_v2/secondary_tiled/results/construction-exclusive/metrics",
      "selection": "maximum"
    },
    "complete_reference": {
      "metrics": {
        "AP": 0.5181880033706425,
        "AP50": 0.9367471310303981,
        "recall_at_threshold": 0.9428172942817294
      },
      "method": "complete_reference_object_crops",
      "seed": 20260917,
      "steps": 6000,
      "resolution": 512,
      "passes": 1,
      "note": "Full-frame inference with NMS",
      "source": "/construction_crop_tiled_interaction/runs/3/full_frame_nms_metrics",
      "selection": "minimum"
    }
  },
  "pawns": {
    "naive": {
      "metrics": {
        "AP": 0.6139736863347561,
        "AP50": 0.826548835275989,
        "recall_at_threshold": 0.5850622406639004
      },
      "method": "naive",
      "seed": 20260918,
      "steps": 2000,
      "resolution": 384,
      "passes": 1,
      "note": "",
      "source": "/tasks/pawns/runs/1/metrics",
      "selection": "minimum"
    },
    "aware": {
      "metrics": {
        "AP": 0.7919867039046043,
        "AP50": 1,
        "recall_at_threshold": 1
      },
      "method": "aware_augmented_512",
      "seed": 20260919,
      "steps": 4000,
      "resolution": 512,
      "passes": 1,
      "note": "",
      "source": "/tasks/pawns/runs/14/metrics",
      "selection": "maximum"
    },
    "complete_reference": {
      "metrics": {
        "AP": 0.7771739747361326,
        "AP50": 1,
        "recall_at_threshold": 1
      },
      "method": "complete_reference",
      "seed": 20260918,
      "steps": 2000,
      "resolution": 384,
      "passes": 1,
      "note": "",
      "source": "/tasks/pawns/runs/7/metrics",
      "selection": "minimum"
    }
  }
};

export function extremaBenchmark(data, key) {
  const matched = matchedBenchmark(data, key);
  const selected = recordedExtrema[key];
  if (!matched || !selected) return null;
  const descriptions = {
    naive: 'Ordinary RF-DETR training on incomplete annotations.',
    aware: 'Coverage-aware training and recorded follow-up experiments.',
    complete_reference: 'RF-DETR trained with all available annotations.',
  };
  const rows = ['naive', 'aware', 'complete_reference'].map(role => ({
    role, ...modelRoles[role], description: descriptions[role],
    metrics: selected[role].metrics, selected: selected[role],
  }));
  return {...matched, rows, gain: 100*(rows[1].metrics.AP-rows[0].metrics.AP),
    gap: 100*(rows[2].metrics.AP-rows[1].metrics.AP)};
}
