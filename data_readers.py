import os
from tqdm import tqdm


class SimpleDirReader(object):
    '''
    Simple data preprocessor. Expects dataset to be in a data_path directory
    in format of:
        data_path/
            entity_1/
                img/
                    images (optional)
                text
                    text data (optional)
            entity_2/
                img/
                    images (optional)
                text
                    text data (optional)
            [...]
            entity_n/
                img/
                    images (optional)
                text
                    text data (optional)
    returns list of dictionaries with path to a given entity, path to its
    images and texts
    '''
    def __init__(self, reserved_paths=['index']):
        self.reserved_paths = reserved_paths

    def process_dir(self, data_path, verbose=True):
        data = []
        paths = os.listdir(data_path)
        if verbose:
            paths = tqdm(paths)
        for path in paths:
            if path in self.reserved_paths:
                continue

            entity_path = os.path.join(data_path, path)
            if not os.path.isdir(entity_path):
                continue

            entity = {
                'path': entity_path,
                'img': [],
                'text': []
            }

            img_path = os.path.join(entity_path, 'img')
            text_path = os.path.join(entity_path, 'text')
            if os.path.isdir(img_path):
                entity['img'] = [os.path.join(img_path, fname)
                                 for fname in os.listdir(img_path)]

            if os.path.isdir(text_path):
                entity['text'] = [
                    open(os.path.join(text_path, fname), 'r').read()
                    for fname in os.listdir(text_path)
                ]
            data.append(entity)
        return data

    def process_single(self, entity_path):
        entity = {
            'path': entity_path,
            'img': [],
            'text': []
        }
        if entity_path in self.reserved_paths:
            raise ValueError('This path is reserved by the application.')

        if not os.path.isdir(entity_path):
            raise NotADirectoryError('Provided path is not a directory.')

        img_path = os.path.join(entity_path, 'img')
        text_path = os.path.join(entity_path, 'text')
        if os.path.isdir(img_path):
            entity['img'] = [os.path.join(img_path, fname)
                             for fname in os.listdir(img_path)]

        if os.path.isdir(text_path):
            entity['text'] = [
                open(os.path.join(text_path, fname), 'r').read()
                for fname in os.listdir(text_path)
            ]
        return entity
