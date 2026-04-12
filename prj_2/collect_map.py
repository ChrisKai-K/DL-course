import subprocess

models = ['faster_rcnn_voc','faster_rcnn_coco','ssd_voc','ssd_coco','yolo_voc','yolo_coco','detr_voc','detr_coco']

with open('map_results.txt', 'w') as out:
    for m in models:
        out.write(f'=== {m} ===\n')
        r = subprocess.run(
            ['python', 'Object-Detection-Metrics/pascalvoc.py',
             '-gt', f'results/{m}/groundtruth',
             '-det', f'results/{m}/detections',
             '-t', '0.5', '-np'],
            capture_output=True, text=True
        )
        out.write(r.stdout)
        if r.stderr:
            out.write(r.stderr)
        out.write('\n')

print(open('map_results.txt').read())
