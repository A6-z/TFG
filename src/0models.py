from pathlib import Path

models = {
    'xlm-roberta-base': 'ml',
    'bert-base-multilingual-cased': 'ml',
    'bert-base-spanish-wwm-cased': 'es',
    'bertin-roberta-base-spanish': 'es',
    'bsc-bio-es': 'es',
    'bsc-bio-ehr-es': 'es',
    'roberta-base-biomedical-clinical-es': 'es',
    'bert-base-italian-xxl-cased': 'it',
    'umberto-commoncrawl-cased-v1': 'it',
    'bioBIT': 'it',
    'medBIT': 'it',
    'roberta-base': 'en',
    'bge-large-en-v1.5': 'en',
    'all-mpnet-base-v2': 'en',
    'pubmedbert-base-embeddings': 'en',
    'biomed_roberta_base': 'en',
    'robeczech-base': 'cz',
    'rubert-base-cased': 'ru',
    'bert-base-NER-Russian': 'ru',
    'RuBioRoBERTa': 'ru',
    'robbert-v2-dutch-base': 'nl',
    'MedRoBERTa.nl': 'nl',
    'bert-base-romanian-cased-v1': 'ro',
    'RoBERT-large': 'ro',
    'bert-base-swedish-cased': 'sw',
    'bert-base-swedish-cased-ner': 'sw',
    'clinical-kb-bert': 'sw'
}

model_dir = Path('/gpfs/projects/bsc14/mouhida/models/tfg')

for model in models.keys():
    print((model_dir / model).exists())