from .tatoeba import load_tatoeba_dataset

fetch_handlers = {
    'tatoeba': load_tatoeba_dataset,
}
