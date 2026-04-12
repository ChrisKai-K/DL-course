import subprocess
import os

models = ['faster_rcnn_voc','faster_rcnn_coco','ssd_voc','ssd_coco','yolo_voc','yolo_coco','detr_voc','detr_coco']
base = os.path.dirname(os.path.abspath(__file__))

with open(os.path.join(base, 'map_results.txt'), 'w') as out:
    for m in models:
        out.write(f'=== {m} ===\n')
        r = subprocess.run(
            ['python', 'pascalvoc.py',
             '-gt', os.path.join(base, 'results', m, 'groundtruth'),
             '-det', os.path.join(base, 'results', m, 'detections'),
             '-t', '0.5', '-np'],
            capture_output=True, text=True,
            cwd=os.path.join(base, 'Object-Detection-Metrics')
        )
        out.write(r.stdout)
        if r.stderr:
            out.write(r.stderr)
        out.write('\n')

print(open(os.path.join(base, 'map_results.txt')).read())
