// A COCO bbox evaluator that mirrors pycocotools COCOeval closely enough to
// reproduce this project's published metrics exactly, so a visitor's browser can
// recompute a score instead of trusting a number baked into the page.
//
// Deliberately narrow: area range "all" and maxDets=100 only, and no iscrowd
// handling. Those are the only settings the published metrics use, and the
// reference labels contain no crowd or ignore annotations. `assertSupported`
// refuses anything outside that envelope rather than quietly scoring it wrong.

// Built the way numpy's linspace builds them, so the thresholds match bit for bit.
function linspace(start, stop, count) {
  const step = (stop - start) / (count - 1), out = new Float64Array(count);
  for (let i = 0; i < count; i += 1) out[i] = start + i * step;
  out[count - 1] = stop;
  return out;
}

export const IOU_THRESHOLDS = linspace(0.5, 0.95, 10);
export const RECALL_THRESHOLDS = linspace(0, 1, 101);
const MAX_DETECTIONS = 100;
const SPACING_1 = 2.220446049250313e-16; // np.spacing(1)
const PAIRWISE_BLOCKSIZE = 128;

// numpy does not add left to right; np.add.reduce sums contiguous float64 data
// pairwise with an 8-way unrolled inner block. Averaging any other way leaves the
// last couple of bits different from the published value, so this mirrors
// numpy's pairwise_sum_DOUBLE exactly.
function pairwiseSum(values, start, count) {
  if (count < 8) {
    let total = 0;
    for (let i = 0; i < count; i += 1) total += values[start + i];
    return total;
  }
  if (count <= PAIRWISE_BLOCKSIZE) {
    const r = [values[start], values[start + 1], values[start + 2], values[start + 3],
               values[start + 4], values[start + 5], values[start + 6], values[start + 7]];
    let i = 8;
    for (; i < count - (count % 8); i += 8) {
      for (let j = 0; j < 8; j += 1) r[j] += values[start + i + j];
    }
    let total = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]));
    for (; i < count; i += 1) total += values[start + i];
    return total;
  }
  let half = Math.floor(count / 2);
  half -= half % 8;
  return pairwiseSum(values, start, half) + pairwiseSum(values, start + half, count - half);
}

function numpyMean(values) {
  return values.length ? pairwiseSum(values, 0, values.length) / values.length : -1;
}

export function assertSupported(groundTruth) {
  for (const annotation of groundTruth.annotations) {
    if (annotation.iscrowd) throw new Error('This evaluator does not handle crowd annotations.');
    if (annotation.ignore) throw new Error('This evaluator does not handle ignored annotations.');
  }
}

// Mirrors the reference bbIou: zero whenever the boxes do not overlap.
function iouMatrix(detections, groundTruths) {
  const rows = detections.length, columns = groundTruths.length;
  const out = new Float64Array(rows * columns);
  for (let g = 0; g < columns; g += 1) {
    const [gx, gy, gw, gh] = groundTruths[g].bbox, gArea = gw * gh;
    for (let d = 0; d < rows; d += 1) {
      const [dx, dy, dw, dh] = detections[d].bbox;
      const w = Math.min(dx + dw, gx + gw) - Math.max(dx, gx);
      if (w <= 0) continue;
      const h = Math.min(dy + dh, gy + gh) - Math.max(dy, gy);
      if (h <= 0) continue;
      const intersection = w * h;
      out[d * columns + g] = intersection / (dw * dh + gArea - intersection);
    }
  }
  return {values: out, columns};
}

// Stable descending sort by score, matching numpy's argsort(-scores, kind='mergesort').
function byScoreDescending(items) {
  return items.map((item, index) => ({item, index}))
    .sort((a, b) => (b.item.score - a.item.score) || (a.index - b.index))
    .map(entry => entry.item);
}

function evaluateImage(groundTruths, detections) {
  if (!groundTruths.length && !detections.length) return null;
  const ordered = byScoreDescending(detections).slice(0, MAX_DETECTIONS);
  const {values: ious, columns} = iouMatrix(ordered, groundTruths);
  const thresholds = IOU_THRESHOLDS.length;
  const matches = new Int8Array(thresholds * ordered.length);
  const claimed = new Int8Array(thresholds * groundTruths.length);
  for (let t = 0; t < thresholds; t += 1) {
    for (let d = 0; d < ordered.length; d += 1) {
      let best = Math.min(IOU_THRESHOLDS[t], 1 - 1e-10), match = -1;
      for (let g = 0; g < groundTruths.length; g += 1) {
        if (claimed[t * groundTruths.length + g]) continue;
        if (columns === 0 || ious[d * columns + g] < best) continue;
        best = ious[d * columns + g];
        match = g;
      }
      if (match === -1) continue;
      matches[t * ordered.length + d] = 1;
      claimed[t * groundTruths.length + match] = 1;
    }
  }
  return {matches, scores: ordered.map(d => d.score), detections: ordered.length,
          positives: groundTruths.length};
}

function groupBy(items, key) {
  const map = new Map();
  for (const item of items) {
    const bucket = key(item);
    if (!map.has(bucket)) map.set(bucket, []);
    map.get(bucket).push(item);
  }
  return map;
}

/**
 * Recompute the published metric set from raw predictions and reference labels.
 * `onProgress` is called with a 0..1 fraction so a caller can render a live count.
 */
export function evaluate(predictions, groundTruth, {threshold = 0.25, onProgress} = {}) {
  assertSupported(groundTruth);
  const categories = [...groundTruth.categories].sort((a, b) => a.id - b.id);
  const imageIds = groundTruth.images.map(image => image.id).slice().sort((a, b) => a - b);
  const gtByImage = groupBy(groundTruth.annotations, a => a.image_id);
  const dtByImage = groupBy(predictions, p => p.image_id);

  const thresholds = IOU_THRESHOLDS.length, recalls = RECALL_THRESHOLDS.length;
  const precision = new Float64Array(thresholds * recalls * categories.length).fill(-1);
  const recall = new Float64Array(thresholds * categories.length).fill(-1);

  categories.forEach((category, k) => {
    const perImage = [];
    for (const imageId of imageIds) {
      const gts = (gtByImage.get(imageId) ?? []).filter(a => a.category_id === category.id);
      const dts = (dtByImage.get(imageId) ?? []).filter(p => p.category_id === category.id);
      const result = evaluateImage(gts, dts);
      if (result) perImage.push(result);
    }
    const positives = perImage.reduce((sum, entry) => sum + entry.positives, 0);
    if (!positives) return;

    // Pool every detection for this class, then rank across the whole split.
    const pooled = [];
    for (const entry of perImage) {
      for (let d = 0; d < entry.detections; d += 1) pooled.push({entry, d});
    }
    pooled.sort((a, b) => (b.entry.scores[b.d] - a.entry.scores[a.d]));
    for (let t = 0; t < thresholds; t += 1) {
      const count = pooled.length;
      const rc = new Float64Array(count), pr = new Float64Array(count);
      let tp = 0, fp = 0;
      for (let i = 0; i < count; i += 1) {
        const {entry, d} = pooled[i];
        if (entry.matches[t * entry.detections + d]) tp += 1; else fp += 1;
        rc[i] = tp / positives;
        pr[i] = tp / (fp + tp + SPACING_1);
      }
      recall[t * categories.length + k] = count ? rc[count - 1] : 0;
      // Precision envelope: make it monotonically decreasing from the right.
      for (let i = count - 1; i > 0; i -= 1) if (pr[i] > pr[i - 1]) pr[i - 1] = pr[i];
      // np.searchsorted(rc, RECALL_THRESHOLDS, side='left'); pycocotools lets the
      // resulting IndexError stop the loop, leaving the rest of the curve at zero.
      // Every threshold for a class with ground truth gets a value: pycocotools
      // starts the curve at zero and only overwrites the part it can reach, so the
      // unreachable tail stays 0 and is averaged in. Leaving it at -1 would drop it
      // from the mean and inflate AP badly at the strict IoU thresholds.
      for (let r = 0; r < recalls; r += 1) precision[(t * recalls + r) * categories.length + k] = 0;
      let cursor = 0;
      for (let r = 0; r < recalls; r += 1) {
        while (cursor < count && rc[cursor] < RECALL_THRESHOLDS[r]) cursor += 1;
        if (cursor >= count) break;
        precision[(t * recalls + r) * categories.length + k] = pr[cursor];
      }
    }
    onProgress?.((k + 1) / categories.length);
  });

  const sliceAt = t => {
    const out = [];
    for (let r = 0; r < recalls; r += 1) {
      for (let k = 0; k < categories.length; k += 1) {
        const value = precision[(t * recalls + r) * categories.length + k];
        if (value > -1) out.push(value);
      }
    }
    return out;
  };
  const everything = [];
  for (let t = 0; t < thresholds; t += 1) everything.push(...sliceAt(t));

  const perClass = {};
  categories.forEach((category, k) => {
    const values = [];
    for (let t = 0; t < thresholds; t += 1) {
      for (let r = 0; r < recalls; r += 1) {
        const value = precision[(t * recalls + r) * categories.length + k];
        if (value >= 0) values.push(value);
      }
    }
    perClass[category.name] = values.length ? numpyMean(values) : null;
  });

  const recallValues = [];
  for (let i = 0; i < recall.length; i += 1) if (recall[i] > -1) recallValues.push(recall[i]);

  return {
    AP: numpyMean(everything), AP50: numpyMean(sliceAt(0)), AP75: numpyMean(sliceAt(5)),
    AR100: numpyMean(recallValues), per_class_AP: perClass,
    ...operatingPoint(predictions, groundTruth, threshold),
  };
}

// The fixed-threshold companion numbers the site shows next to AP.
function operatingPoint(predictions, groundTruth, threshold) {
  const observed = new Set(groundTruth.annotations.map(a => a.image_id));
  const negatives = new Set(groundTruth.images.map(i => i.id).filter(id => !observed.has(id)));
  const active = predictions.filter(p => p.score >= threshold);
  const activeByImage = groupBy(active, p => p.image_id);
  const gtByImage = groupBy(groundTruth.annotations, a => a.image_id);
  let truePositive = 0, falsePositive = 0;
  for (const image of groundTruth.images) {
    const ground = gtByImage.get(image.id) ?? [];
    const claimed = new Set();
    for (const prediction of byScoreDescending(activeByImage.get(image.id) ?? [])) {
      let bestIou = 0, bestIndex = -1;
      ground.forEach((annotation, index) => {
        if (claimed.has(index) || prediction.category_id !== annotation.category_id) return;
        const [ax, ay, aw, ah] = prediction.bbox, [bx, by, bw, bh] = annotation.bbox;
        const intersection = Math.max(0, Math.min(ax + aw, bx + bw) - Math.max(ax, bx))
                           * Math.max(0, Math.min(ay + ah, by + bh) - Math.max(ay, by));
        const value = intersection / Math.max(aw * ah + bw * bh - intersection, 1e-12);
        // Matches Python's max((iou, index)): highest IoU, ties go to the later box.
        if (value > bestIou || (value === bestIou && index > bestIndex)) { bestIou = value; bestIndex = index; }
      });
      if (bestIou >= 0.5) { truePositive += 1; claimed.add(bestIndex); } else falsePositive += 1;
    }
  }
  const support = {};
  for (const annotation of groundTruth.annotations) {
    const key = String(annotation.category_id);
    support[key] = (support[key] ?? 0) + 1;
  }
  return {
    precision_at_threshold: truePositive / Math.max(1, truePositive + falsePositive),
    recall_at_threshold: truePositive / Math.max(1, groundTruth.annotations.length),
    false_positives_per_image: falsePositive / Math.max(1, groundTruth.images.length),
    negative_images: negatives.size,
    negative_image_false_positives_per_image:
      active.filter(p => negatives.has(p.image_id)).length / Math.max(1, negatives.size),
    images: groundTruth.images.length, boxes: groundTruth.annotations.length, class_support: support,
  };
}

/** Decode the compact CVB1 prediction blob the build script writes. */
export function decodePredictions(buffer) {
  const view = new DataView(buffer);
  const magic = String.fromCharCode(view.getUint8(0), view.getUint8(1), view.getUint8(2), view.getUint8(3));
  if (magic !== 'CVB1') throw new Error('Unrecognised prediction bundle.');
  if (view.getUint32(4, true) !== 1) throw new Error('Unsupported prediction bundle version.');
  const count = view.getUint32(8, true);
  const imageIds = new Int32Array(buffer, 16, count);
  const categoryIds = new Int32Array(buffer, 16 + 4 * count, count);
  const scores = new Float32Array(buffer, 16 + 8 * count, count);
  const boxes = new Float32Array(buffer, 16 + 12 * count, 4 * count);
  const out = new Array(count);
  for (let i = 0; i < count; i += 1) {
    out[i] = {image_id: imageIds[i], category_id: categoryIds[i], score: scores[i],
              bbox: [boxes[4 * i], boxes[4 * i + 1], boxes[4 * i + 2], boxes[4 * i + 3]]};
  }
  return out;
}

/** SHA-256 of the exact bytes that were scored, as lowercase hex. */
export async function sha256Hex(buffer) {
  const digest = await crypto.subtle.digest('SHA-256', buffer);
  return [...new Uint8Array(digest)].map(b => b.toString(16).padStart(2, '0')).join('');
}
