"""Bind the prediction gallery to the same matched cohorts the rest of the site shows.

The gallery used to illustrate the recorded extrema, which mixed recipes. Once the
benchmark pages moved to matched cohorts, the gallery was showing a different pair of
models for the same dataset while claiming to show the same runs. It now reads the
verification manifest, features one cohort per dataset, and draws its boxes with a single
seed from that cohort -- so the score on this page is the score on every other page.
"""
import hashlib
import json
import math
from pathlib import Path

from coveragecv.artifacts import file_digest, read_json, verify

ROOT = Path(__file__).resolve().parents[1]
STATIC = ROOT / 'src/coveragecv/workbench/static'
# One cohort per dataset, named by its verification-manifest key. The gallery must feature
# a cohort the Verify page can recompute, or the pictures and the score come apart again.
FEATURED = {'pawns': 'pawns-base', 'all-pieces': 'all-pieces-base', 'construction': 'construction'}
MANIFEST = ROOT / 'src/coveragecv/workbench/static/verify/manifest.json'


def cohort_sources():
    """Resolve each featured cohort to one seed's run folders, from the manifest."""
    manifest = read_json(MANIFEST)
    sources, cohorts = {}, {}
    for task, key in FEATURED.items():
        cohort = next((entry for entry in manifest['cohorts'] if entry['key'] == key), None)
        if cohort is None:
            raise ValueError(f'{task}: cohort {key} is absent from the verification manifest')
        seed = cohort['seeds'][0]
        arms, means = {}, {}
        for role in ('naive', 'aware', 'complete_reference'):
            matching = [run for run in cohort['runs'] if run['role'] == role]
            values = [run['expected'] for run in matching]
            means[role] = {name: sum(entry[name] for entry in values) / len(values)
                           for name in ('AP', 'AP50', 'recall_at_threshold')}
            run = next(entry for entry in matching if entry['seed'] == seed)
            arms[role] = (str(Path(run['source_evaluation_path']).parent), 'evaluation.json', run)
        sources[task] = arms
        cohorts[task] = {'key': key, 'name': cohort['name'], 'recipe': cohort['recipe'],
                         'seed': seed, 'seeds': cohort['seeds'], 'steps': cohort['steps'],
                         'resolution': cohort['resolution'], 'means': means}
    return sources, cohorts


def main():
    SOURCES, COHORTS = cohort_sources()
    ledger = read_json(ROOT / 'docs/RESULTS.json')
    snapshot = read_json(ROOT / 'public-demo/data/snapshot.json')['routes']
    routes, selected, receipts = {}, {}, []
    for task, sources in SOURCES.items():
        digest = ledger['tasks'][task]['reference_bundle_digest']
        template_path, examples = next((path, value) for path, value in snapshot.items()
                                      if path.endswith('/examples') and value['bundle_digest'] == digest)
        old_id = template_path.split('/')[2]
        binding = read_json(ROOT / 'artifacts/research_v2/protocol.json')['datasets'][task]
        reference = Path(binding['local_reference'])
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
        cohort = COHORTS[task]
        for arm, (folder_name, evaluation_name, manifest_run) in sources.items():
            folder = ROOT / folder_name
            evaluation_path = folder / evaluation_name
            run, evaluation = read_json(folder / 'run.json'), read_json(evaluation_path)
            chosen = {'metrics': manifest_run['expected'], 'steps': manifest_run['steps'],
                      'resolution': manifest_run['resolution'], 'seed': manifest_run['seed'],
                      'method': manifest_run['method'], 'passes': 1,
                      'cohort': cohort['key'], 'cohort_metrics': cohort['means'][arm]}
            sha = file_digest(folder / 'detector.pt')
            if sha != run['detector_sha256'] or sha != evaluation['checkpoint_sha256']:
                raise ValueError(f'{task}/{arm}: checkpoint identity mismatch')
            if evaluation['bundle_digest'] != digest or evaluation['split'] != 'valid':
                raise ValueError(f'{task}/{arm}: wrong evaluation cohort')
            for metric, value in chosen['metrics'].items():
                if evaluation['metrics'][metric] != value:
                    raise ValueError(f'{task}/{arm}: displayed score differs from saved predictions')
            if manifest_run['checkpoint_sha256'] != sha:
                raise ValueError(f'{task}/{arm}: manifest checkpoint differs from the gallery checkpoint')
            for row in evaluation['predictions']:
                if (row['image_id'] not in image_ids or not 1 <= row['category_id'] <= len(examples['classes'])
                        or not math.isfinite(row['score']) or not 0 <= row['score'] <= 1
                        or len(row['bbox']) != 4 or not all(math.isfinite(x) for x in row['bbox'])):
                    raise ValueError('Invalid prediction payload')
            arms[arm] = {
                # The headline is the cohort's matched, seed-averaged score, identical to the
                # Benchmarks and Verify pages. The single run that drew the boxes is beside it.
                'metrics': cohort['means'][arm],
                'run_metrics': evaluation['metrics'],
                'run': {key: run[key] for key in ('arm', 'seed', 'steps', 'device', 'batch')},
                'selection': chosen,
                'checkpoint_sha256': sha,
                'evaluation_sha256': file_digest(evaluation_path),
            }
            arms[arm]['run'].update(total_training_steps=chosen['steps'], resolution=chosen['resolution'])
            for image in images:
                predictions[image['id']][arm] = [row for row in evaluation['predictions'] if row['image_id'] == image['id']]
            receipts.append({'task': task, 'arm': arm, 'cohort': cohort['key'],
                             'cohort_AP': cohort['means'][arm]['AP'], 'AP': chosen['metrics']['AP'],
                             'checkpoint_sha256': sha, 'evaluation_sha256': file_digest(evaluation_path)})
        selected[task] = {'id': selection_id, 'kind': 'recorded_selection', 'status': 'completed',
                          'selection_kind': 'cohort', 'cohort': cohort, 'result': {'arms': arms}}
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
