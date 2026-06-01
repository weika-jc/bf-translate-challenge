from .tatoeba import load_tatoeba_dataset
from .subtitle import load_subtitle_dataset
from .mtnt import load_mtnt_dataset

ALL_LANGUAGES = ['en', 'es', 'ja', 'de', 'fr', 'nl', 'da', 'no', 'it']

fetch_handlers = {
    'tatoeba': {
        'handler': load_tatoeba_dataset,
        'languages': ALL_LANGUAGES,
        'need_filter': True,
    },
    'subtitle': {
        'handler': load_subtitle_dataset,
        'languages': ALL_LANGUAGES,
        'need_filter': True,
    },
    'mtnt': {
        'handler': load_mtnt_dataset,
        'languages': ['en', 'ja', 'fr'],
        'need_filter': False,
    },
}
