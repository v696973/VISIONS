# -*- coding: utf-8 -*-
import os
from tqdm import tqdm
from utils import img_or_text


class DataReader(object):
    '''
    Base class for data readers.
    '''
    def __init__(self, reserved_paths=['index']):
        self.reserved_paths = reserved_paths


class SimpleDirReader(DataReader):
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
        super().__init__(reserved_paths)

    def process_data(self, data_path, verbose=True):
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
        if not os.path.exists(entity_path):
            raise ValueError('There is no entity under provided path.')

        if entity_path in self.reserved_paths:
            raise ValueError('This path is reserved by the application.')

        if not os.path.isdir(entity_path):
            raise NotADirectoryError('Provided path is not a directory.')

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
        return entity


class FlatDirReader(DataReader):
    '''
    This data reader simply reads a flat directory, which can contain both
    text files and images. The reader does its best at identifying image
    and text files, using simple heuristics, therefore it's best to keep texts
    in plain txt files when processing them using this data reader.
    '''
    def __init__(self):
        super().__init__()

    def process_data(self, data_path, verbose=True):
        data = []
        paths = os.listdir(data_path)
        if verbose:
            paths = tqdm(paths)
        for path in paths:
            if path in self.reserved_paths and not os.path.isfile(path):
                continue

            entity_path = os.path.join(data_path, path)

            entity_type = img_or_text(entity_path)

            if entity_type == 'img':
                entity = {
                    'path': entity_path,
                    'img': [entity_path],
                    'text': []
                }
            elif entity_type == 'text':
                with open(entity_path, 'r') as f:
                    entity_text = f.read()

                entity = {
                    'path': entity_path,
                    'img': [],
                    'text': [entity_text]
                }
            else:
                continue  # TODO raise warning here, or do proper logging

            data.append(entity)
        return data

    def process_single(self, entity_path):
        if not os.path.exists(entity_path):
            raise ValueError('There is no entity under provided path.')

        if entity_path in self.reserved_paths:
            raise ValueError('Provided path is reserved by the application.')

        if not os.path.isfile(entity_path):
            raise NotADirectoryError('Provided path is not a file.')

        entity_type = img_or_text(entity_path)

        if entity_type == 'img':
            entity = {
                'path': entity_path,
                'img': [entity_path],
                'text': []
            }
        elif entity_type == 'text':
            with open(entity_path, 'r') as f:
                entity_text = f.read()

            entity = {
                'path': entity_path,
                'img': [],
                'text': [entity_text]
            }
        else:
            pass  # TODO raise warning here, or do proper logging

        return entity
