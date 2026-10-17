"""Explicit, read-only hosted-model checks. Uses the saved key; never prints it.

PYTHONPATH=src python scripts/check_studio_nvidia.py --output /tmp/nvidia-check.json
Synthetic profiles test instruction following, not clinical phenotype accuracy.
No project data is sent and no Studio commands are executed.
"""

import argparse
import base64
import io
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from spacemap_assistant.model_client import NVIDIA_MODEL_CATALOG, NvidiaModelClient
from spacemap_gui.studio.chat import read_reply


def tasks():
    # Explicit artificial reference vectors: reproducible expected answers, no
    # unsupported intensity thresholds or clinical ground truth assumptions.
    markers = ['CD3e', 'CD4', 'CD8', 'FoxP3', 'CD20', 'PanCK', 'CD31', 'CD68']
    references = {
        'CD4': [8, 8, 0, 0, 0, 0, 0, 0], 'CD8': [8, 0, 8, 0, 0, 0, 0, 0],
        'Treg': [8, 8, 0, 8, 0, 0, 0, 0], 'B': [0, 0, 0, 0, 8, 0, 0, 0],
        'Epithelia': [0, 0, 0, 0, 0, 8, 0, 0], 'Endothelia': [0, 0, 0, 0, 0, 0, 8, 0],
        'Macrophage': [0, 0, 0, 0, 0, 0, 0, 8],
    }
    order = ['Epithelia', 'Treg', 'B', 'CD8', 'Macrophage', 'CD4', 'Endothelia']
    profiles = {f'Cluster {i}': references[name] for i, name in enumerate(order)}
    profiles.update({'Cluster 7': [0]*8, 'Cluster 8': [4]*8, 'Cluster 9': [None]*8})
    expected = {f'Cluster {i}': name for i, name in enumerate(order + ['Others']*3)}
    prompt = (
        'This is an artificial software test, not tissue data. Match each cluster to the supplied reference vectors. '
        'Exact matches receive that reference name. All-zero, mixed, or missing profiles receive Others. '
        'Do not infer positivity or use outside medical knowledge. Return only JSON '
        '{"say":"Brief evidence and uncertainty","action":{"name":"name_clusters",'
        '"args":{"job_id":"synthetic-check","names":{"Cluster 0":"a reference name"}}}}. '
        'Include all ten clusters; propose only, never claim to have saved changes. DATA:\n'
        + json.dumps({'markers': markers, 'references': references, 'profiles': profiles})
    )
    image = Image.new('RGB', (600, 200), 'white')
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype('DejaVuSans.ttf', 32)
    draw.text((25, 20), 'Sample Q7', font=font, fill='black')
    draw.text((25, 80), 'CD3e = 73', font=font, fill='black')
    draw.text((310, 80), 'CD20 = 18', font=font, fill='black')
    out = io.BytesIO()
    image.save(out, format='PNG')
    url = 'data:image/png;base64,' + base64.b64encode(out.getvalue()).decode()
    return [
        ('connection_1', [{'role': 'user', 'content': 'Return only JSON {"status":"ready"}.'}], {'status': 'ready'}),
        ('connection_2', [{'role': 'user', 'content': 'Return only JSON {"status":"ready"}.'}], {'status': 'ready'}),
        ('profiles', [{'role': 'user', 'content': prompt}], expected),
        ('image', [{'role': 'user', 'content': [
            {'type': 'text', 'text': 'Read the supplied test image. Return only JSON with keys sample, CD3e and CD20; transcribe the sample and numbers exactly.'},
            {'type': 'image_url', 'image_url': {'url': url}},
        ]}], {'sample': 'Q7', 'CD3e': 73, 'CD20': 18}),
    ]


def check(model, mode):
    results = []
    for name, messages, expected in tasks():
        if name == 'image' and not NVIDIA_MODEL_CATALOG[model]['supports_images']:
            results.append({'test': name, 'skipped': 'Text-only model'})
            continue
        client = NvidiaModelClient(model_id=model, timeout_seconds=45)
        client.require_complete = True
        client.max_response_seconds = 180
        start = time.monotonic()
        row = {'test': name, 'first_answer_s': None, 'first_activity_s': None}

        def progress(event):
            if row['first_activity_s'] is None:
                row['first_activity_s'] = round(time.monotonic()-start, 2)
            if time.monotonic()-start > 180:
                raise TimeoutError('Exceeded 180-second check budget')

        options = {'max_tokens': 8192, 'temperature': 1.0,
                   'enable_thinking': mode != 'fast', 'on_progress': progress}
        if model == 'meta/muse-glimmer-30b':
            options['reasoning_strength'] = 'low' if mode == 'fast' else 'medium'
        try:
            pieces = []
            for piece in client.stream_complete(messages, **options):
                progress(piece)
                if row['first_answer_s'] is None:
                    row['first_answer_s'] = round(time.monotonic()-start, 2)
                pieces.append(piece)
            raw = ''.join(pieces).strip()
            row['answer'] = raw
            if name == 'profiles':
                _, action, _ = read_reply(raw)
                names = ((action or {}).get('args') or {}).get('names', {})
                row['valid_action'] = bool(action and action['name'] == 'name_clusters' and action['args'].get('job_id') == 'synthetic-check')
                row['correct'] = sum(names.get(k) == v for k, v in expected.items())
                row['total'] = len(expected)
                row['passed'] = row['valid_action'] and names == expected
            else:
                try:
                    value = json.loads(raw.removeprefix('```json').removeprefix('```').removesuffix('```').strip())
                except ValueError:
                    value = None
                row['exact_json_match'] = value == expected
                if name == 'image' and isinstance(value, dict):
                    # The task asks for transcription, not a JSON number type.
                    # Accept the printed "Sample" prefix and quoted numbers.
                    value = {str(k): str(v).removeprefix('Sample ') for k, v in value.items()}
                    row['passed'] = value == {k: str(v) for k, v in expected.items()}
                else:
                    row['passed'] = value == expected
        except Exception as error:
            row.update(passed=False, error=str(error))
        row['total_s'] = round(time.monotonic()-start, 2)
        row['completion'] = getattr(client, 'last_completion', {})
        results.append(row)
        print(json.dumps({'model': model, 'mode': mode, **{k:v for k,v in row.items() if k != 'answer'}}), flush=True)
    return {'model': model, 'mode': mode, 'tests': results}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--mode', choices=['balanced', 'fast'], default='balanced')
    parser.add_argument('--model', choices=list(NVIDIA_MODEL_CATALOG), action='append')
    args = parser.parse_args()
    report = {'checked_at': datetime.now(timezone.utc).isoformat(), 'models': [],
              'scope': 'Synthetic software checks; not a clinical accuracy benchmark. No project mutations.',
              'settings': {'max_tokens': 8192, 'temperature': 1.0, 'socket_timeout_seconds': 45}}
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(check, model, args.mode) for model in (args.model or NVIDIA_MODEL_CATALOG)]
        for future in as_completed(futures):
            report['models'].append(future.result())
            args.output.write_text(json.dumps(report, indent=2) + '\n')
