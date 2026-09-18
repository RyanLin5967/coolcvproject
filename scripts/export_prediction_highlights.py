"""Bind the benchmark extrema gallery to the exact saved checkpoint evaluations."""
import hashlib
import json
import math
from pathlib import Path
from urllib.request import urlopen

from coveragecv.artifacts import file_digest, read_json, verify

ROOT = Path(__file__).resolve().parents[1]
STATIC = ROOT / 'src/coveragecv/workbench/static'
# Explicit evidence files: changing scores alone cannot change which predictions are exported.
SOURCES = {
    'pawns': {
        'naive': ('artifacts/gpu/pawns/20260918/naive', 'evaluation.json'),
        'aware': ('artifacts/improved/pawns/20260919/aware_augmented_512', 'evaluation.json'),
        'complete_reference': ('artifacts/gpu/pawns/20260918/complete_reference', 'evaluation.json'),
    },
    'all-pieces': {
        'naive': ('artifacts/gpu/all-pieces/20260919/naive', 'evaluation.json'),
        'aware': ('artifacts/capacity/all-pieces/20260917/aware_large_704_ema', 'evaluation.json'),
        'complete_reference': ('artifacts/gpu/all-pieces/20260917/complete_reference', 'evaluation.json'),
    },
    'construction': {
        'naive': ('artifacts/object-crops/construction/20260917/naive_object_crops',
                  'artifacts/crop-tiled/construction/20260917/naive_object_crops/tiled.json'),
        'aware': ('artifacts/acquisition/construction/20260917/guided',
                  'artifacts/acquisition-tiled/construction/20260917/guided/size_gated.json'),
        'complete_reference': ('artifacts/object-crops/construction/20260917/complete_reference_object_crops',
                               'artifacts/crop-tiled/construction/20260917/complete_reference_object_crops/full_frame_nms.json'),
    },
}


def main():
    text = (STATIC / 'benchmark-view.js').read_text().split('export const recordedExtrema = ', 1)[1]
    selections, _ = json.JSONDecoder().raw_decode(text)
    ledger = read_json(ROOT / 'docs/RESULTS.json')
    snapshot = read_json(ROOT / 'public-demo/data/snapshot.json')['routes']
    routes, selected, receipts = {}, {}, []
    for task, sources in SOURCES.items():
        digest = ledger['tasks'][task]['reference_bundle_digest']
        template_path, examples = next((path, value) for path, value in snapshot.items()
                                      if path.endswith('/examples') and value['bundle_digest'] == digest)
        old_id = template_path.split('/')[2]
        with urlopen(f'http://127.0.0.1:8765/api/jobs/{old_id}', timeout=30) as response:
            old_job = json.load(response)
        reference = Path(old_job['result']['reference_bundle'])
        if verify(reference)['digest'] != digest:
            raise ValueError('Reference bundle failed verification')
        full = read_json(reference / 'splits/valid.coco.json')
        image_ids = {image['id'] for image in full['images']}
        count = len(image_ids)
        images = []
        for image in examples['images']:
            payload = (ROOT / 'public-demo' / image['image_url'].lstrip('/')).read_bytes()
            if hashlib.sha256(payload).hexdigest() != Path(image['file_name']).stem:
                raise ValueError('Gallery image content has changed')
            images.append({**image, 'local_image_url': f'/api/jobs/{old_id}/images/{image["id"]}'})
        selection_id = f'highlights-{task}'
        arms, predictions = {}, {image['id']: {} for image in images}
        for arm, (folder_name, evaluation_name) in sources.items():
            folder = ROOT / folder_name
            evaluation_path = folder / evaluation_name if evaluation_name == 'evaluation.json' else ROOT / evaluation_name
            run, evaluation = read_json(folder / 'run.json'), read_json(evaluation_path)
            chosen = selections[task][arm]
            sha = file_digest(folder / 'detector.pt')
            if sha != run['detector_sha256'] or sha != evaluation['checkpoint_sha256']:
                raise ValueError(f'{task}/{arm}: checkpoint identity mismatch')
            if evaluation['bundle_digest'] != digest or evaluation['split'] != 'valid':
                raise ValueError(f'{task}/{arm}: wrong evaluation cohort')
            for metric, value in chosen['metrics'].items():
                if evaluation['metrics'][metric] != value:
                    raise ValueError(f'{task}/{arm}: displayed score differs from saved predictions')
            for row in evaluation['predictions']:
                if (row['image_id'] not in image_ids or not 1 <= row['category_id'] <= len(examples['classes'])
                        or not math.isfinite(row['score']) or not 0 <= row['score'] <= 1
                        or len(row['bbox']) != 4 or not all(math.isfinite(x) for x in row['bbox'])):
                    raise ValueError('Invalid prediction payload')
            arms[arm] = {
                'metrics': evaluation['metrics'],
                'run': {key: run[key] for key in ('arm', 'seed', 'steps', 'device', 'batch')},
                'selection': chosen,
                'checkpoint_sha256': sha,
                'evaluation_sha256': file_digest(evaluation_path),
            }
            arms[arm]['run'].update(total_training_steps=chosen['steps'], resolution=chosen['resolution'])
            for image in images:
                predictions[image['id']][arm] = [row for row in evaluation['predictions'] if row['image_id'] == image['id']]
            receipts.append({'task': task, 'arm': arm, 'AP': chosen['metrics']['AP'],
                             'checkpoint_sha256': sha, 'evaluation_sha256': file_digest(evaluation_path)})
        selected[task] = {'id': selection_id, 'kind': 'recorded_selection', 'status': 'completed',
                          'selection_kind': 'extrema', 'result': {'arms': arms}}
        routes[f'/jobs/{selection_id}/examples'] = {
            'bundle_digest': digest, 'classes': examples['classes'], 'images': images,
            'total_validation_images': count,
            'selection': 'Same six evenly spaced validation images; selected before reading model predictions.',
        }
        for image_id, rows in predictions.items():
            routes[f'/jobs/{selection_id}/predictions/{image_id}'] = rows
    output = {'schema_version': 1, 'selections': selected, 'routes': routes,
              'source_results_sha256': file_digest(ROOT / 'docs/RESULTS.json')}
    encoded = json.dumps(output, sort_keys=True, separators=(',', ':'), allow_nan=False) + '\n'
    if any(value in encoded for value in ('/Users/', '/home/', '/root/', 'token_secret', 'api_key')):
        raise ValueError('Private information in public prediction export')
    (STATIC / 'prediction-highlights.json').write_text(encoded)
    print(json.dumps({'bytes': len(encoded.encode()), 'verified_evaluations': receipts}, indent=2))


if __name__ == '__main__':
    main()
