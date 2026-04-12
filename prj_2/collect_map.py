import subprocess
import os

models = ['faster_rcnn_voc','faster_rcnn_coco','ssd_voc','ssd_coco','yolo_voc','yolo_coco','detr_voc','detr_coco']
base = os.path.dirname(os.path.abspath(__file__))

with open(os.path.join(base, 'map_results.txt'), 'w') as out:
    for m in models:
        out.write(f'=== {m} ===\n')
        proc = subprocess.Popen(
            ['python', 'pascalvoc.py',
             '-gt', os.path.join(base, 'results', m, 'groundtruth'),
             '-det', os.path.join(base, 'results', m, 'detections'),
             '-t', '0.5', '-np'],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True,
            cwd=os.path.join(base, 'Object-Detection-Metrics')
        )
        stdout, stderr = proc.communicate(input='Y\n')
        out.write(stdout)
        if stderr:
            out.write(stderr)
        out.write('\n')

print(open(os.path.join(base, 'map_results.txt')).read())
