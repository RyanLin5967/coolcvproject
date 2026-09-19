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

// Independently selected AP50:95 extrema from the published results ledger.
// Source ledger SHA-256: e0255eb45de1900bdbcafc28d89c554e71ae6240debe4d79f229ad7359dfd7a4
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
        "AP": 0.5263631559110266,
        "AP50": 0.9062293944689279,
        "recall_at_threshold": 0.9288702928870293
      },
      "method": "native_resolution_coverage",
      "seed": 20260919,
      "steps": 10000,
      "resolution": 640,
      "passes": 5,
      "note": "265 additional published training boxes",
      "source": "/continuous_research/confirmation/cases/construction-aware-scale/validation_tiled_metrics",
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

// Seed-averaged matched comparison, driven by the verification manifest. Rendering the
// benchmark cards from the same source the Verify page scores means every headline number
// on the site is one a visitor can recompute; the extrema below are kept, but demoted to
// a clearly labelled aside, because they mix recipes and are not a matched comparison.
export async function loadVerificationManifest() {
  const response = await fetch('/static/verify/manifest.json');
  if (!response.ok) throw new Error('The verification manifest is not published with this build.');
  return response.json();
}

export function cohortBenchmark(manifest, key) {
  const cohort = manifest.cohorts.find(entry => entry.key === key);
  if (!cohort) return null;
  const roles = ['naive', 'aware', 'complete_reference'];
  const rows = roles.map(role => {
    const runs = cohort.runs.filter(run => run.role === role);
    if (runs.length !== cohort.seeds.length) throw new Error('Benchmark cohort is missing an arm.');
    const metrics = {};
    for (const name of ['AP', 'AP50', 'recall_at_threshold']) {
      const values = runs.map(run => run.expected[name]);
      if (!values.every(value => Number.isFinite(value) && value >= 0 && value <= 1)) {
        throw new Error('Benchmark contains an invalid metric.');
      }
      metrics[name] = values.reduce((sum, value) => sum + value, 0) / values.length;
    }
    return {role, ...modelRoles[role], metrics, runs};
  });
  const labels = manifest.ground_truth[cohort.runs[0].bundle_digest];
  return {...cohort, rows, images: labels.images, boxes: labels.boxes, classes: labels.categories,
          gain: 100 * (rows[1].metrics.AP - rows[0].metrics.AP),
          gap: 100 * (rows[2].metrics.AP - rows[1].metrics.AP)};
}
