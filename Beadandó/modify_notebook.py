import os
import json

try:
    notebook_filename = [f for f in os.listdir('.') if f == 'san_francisco_crime_V7.ipynb'][0]
except IndexError:
    print("Notebook not found in current directory:", os.getcwd())
    print("Files in dir:", os.listdir('.'))
    exit(1)

with open(notebook_filename, 'r', encoding='utf-8') as f:
    nb = json.load(f)

for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source = cell['source']
        if len(source) > 0 and 'KÜLÖN TÉRKÉP GENERÁLÁSA' in source[0]:
            new_source = []
            skip = False
            for line in source:
                if '# Térkép mentése' in line:
                    skip = True
                    new_source.append("# Térkép megjelenítése a notebookban\\n")
                    new_source.append("m_tomorrow\\n")
                    break
                if not skip:
                    new_source.append(line)
            cell['source'] = new_source

with open(notebook_filename, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
    f.write('\\n')

print("Jupyter notebook frissítve!")
