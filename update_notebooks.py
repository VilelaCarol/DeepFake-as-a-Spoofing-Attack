import json
import re

def update_lfw_adaface():
    with open('pipeline_lfw_adaface.ipynb', 'r', encoding='utf-8') as f:
        nb = json.load(f)

    for cell in nb['cells']:
        if cell['cell_type'] == 'markdown':
            new_source = []
            for line in cell['source']:
                line = line.replace('30 swaps', '3000 swaps')
                line = line.replace('28/30', '2783/3000')
                line = line.replace('93.3%', '92.8%')
                line = line.replace('27/30', '2737/3000')
                line = line.replace('90.0%', '91.2%')
                line = line.replace('2/30', '217/3000')
                line = line.replace('3/30', '263/3000')
                line = line.replace('0.6846', '0.6990')
                line = line.replace('0.7565', '0.7541')
                line = line.replace('-0.1167', '-0.1552')
                line = line.replace('0.8232', '0.9238')
                line = line.replace('Python 3.10', 'Python 3.13')
                if '| Nº de Swaps Gerados (LFW) |' in line:
                    line = '| Nº de Swaps Gerados (LFW) | 3000 |\n'
                new_source.append(line)
            cell['source'] = new_source

    with open('pipeline_lfw_adaface.ipynb', 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1)


def update_deepfake():
    with open('pipeline_deepfake.ipynb', 'r', encoding='utf-8') as f:
        nb = json.load(f)

    for cell in nb['cells']:
        if cell['cell_type'] == 'markdown':
            new_source = []
            for line in cell['source']:
                line = line.replace('ASR de 93% no LFW', 'ASR de 92.8% no LFW')
                line = line.replace('ASR caiu para 20% no YTF', 'ASR caiu para 34.3% no YTF')
                
                # Update architecture
                if '│ Vídeo da' in line:
                    line = line.replace('Vídeo da', 'Webcam  ')
                if '│ (.mp4)' in line:
                    line = line.replace('(.mp4)  ', '(Live)  ')
                
                new_source.append(line)
            cell['source'] = new_source

    with open('pipeline_deepfake.ipynb', 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1)

update_lfw_adaface()
update_deepfake()
print("Notebooks updated successfully!")
