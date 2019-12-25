import os
import argparse
import index
from tqdm import tqdm
import data_readers
from img_embedding import MobileNetV2Model
from text_embedding import M_USEModel
from vsepp import VSEPPModel
# TODO: proper logging


def build_index(args):
    img_encoder = MobileNetV2Model()
    text_encoder = M_USEModel()
    vse = VSEPPModel(
        img_encoder,
        text_encoder,
        img_aligner_weights='models/vsepp/512/ep30_dim_512_lr2e-06_img.h5',
        text_aligner_weights='models/vsepp/512/ep30_dim_512_lr2e-06_text.h5'
    )
    if args.index_path:
        index_path = args.index_path
    else:
        index_path = os.path.join(args.data_path, 'index/')

    if args.index_type == 'flat':
        idx = index.FlatIndex()
        if args.append:
            try:
                idx.connect(index_path)
            except FileNotFoundError:
                print('No index found at `{0}`. '
                      'Creating new index...'.format(index_path))
                idx.create(index_path, bucket_size=args.flat_bucket_size)
        else:
            try:
                idx.create(index_path,
                           bucket_size=args.flat_bucket_size,
                           force=args.rebuild)
            except FileExistsError:
                print('Index at `{}` already exists. Use `--append` '
                      'in order to append new entities to it, or `--rebuild` '
                      'in order to rebuild it'.format(index_path))
                raise SystemExit
    else:  # TODO
        raise NotImplementedError('Unknown index type.')

    if args.data_reader_type == 'simple_dir':
        data_reader = data_readers.SimpleDirReader()
    else:  # TODO?
        raise NotImplementedError('Unknown data reader type.')
    data = data_reader.process_dir(args.data_path)
    for entity in tqdm(data):
        # TODO: add config for embedding merge strategies
        entity_vec = vse.embed(entity['img'], entity['text'])
        # entity_vec = text_encoder.embed(entity['text'])[0]
        ids = idx.add_item(entity['path'], entity_vec)
        if ids[0] == -1:
            print('Entity already in index, skipping')


def query(args):
    idx = index.FlatIndex()

    try:
        idx.connect(args.index_path)
    except FileNotFoundError:
        print('No index found at `{0}`. '
              'Use build_index mode in order '
              'to create new index'.format(args.index_path))
        raise SystemExit

    if args.data_reader_type == 'simple_dir':
        data_reader = data_readers.SimpleDirReader()
    else:  # TODO?
        raise NotImplementedError('Unknown data reader type.')

    if args.query_text or args.query_path:
        img_encoder = MobileNetV2Model()
        text_encoder = M_USEModel()
        vse = VSEPPModel(
            img_encoder,
            text_encoder,
            img_aligner_weights='models/vsepp/512/ep30_dim_512_lr2e-06_img.h5',
            text_aligner_weights='models/vsepp/512/ep30_dim_512_lr2e-06_text.h5'  # NOQA
        )
        if args.query_text:
            # TODO: config for merge strategies
            entity = {
                'img': [],
                'text': [args.query_text]
            }
        elif args.query_path:
            entity = data_reader.process_single(args.query_path)

        entity_vec = vse.embed(entity['img'], entity['text'])
        # entity_vec = text_encoder.embed(entity['text'])[0]
        nns = idx.get_nn_by_vec(entity_vec, n=args.n)
        for nn in nns:
            print(nn[0], nn[1])

    # TODO: interactive mode


def main():
    parser = argparse.ArgumentParser(description='')
    subparsers = parser.add_subparsers()
    build_index_parser = subparsers.add_parser(
        'build_index',
        help='build index for a given dataset'
    )
    build_index_parser.add_argument(
        '--data_path',
        required=True,
        help='path to dataset'
    )
    build_index_parser.add_argument('--index_path', help='path to data index')
    build_index_parser.add_argument(
        '--index_type',
        default='flat',
        const='flat',
        nargs='?',
        choices=('flat', 'annoy'),  # TODO annoy index
        help='index type (default: %(default)s)'
    )
    build_index_parser.add_argument(
        '--data_reader_type',
        default='simple_dir',
        const='simple_dir',
        nargs='?',
        choices=('simple_dir',),
        help='index type (default: %(default)s)'
    )
    build_index_parser.add_argument(
        '--flat_bucket_size',
        default=512,
        type=int,
        help='flat index bucket size'
    )
    build_index_parser.add_argument(
        '--rebuild',
        action='store_true',
        help='rebuild index from scratch'
    )
    build_index_parser.add_argument(
        '--append',
        action='store_true',
        help='append new items from data_path to index'
    )
    build_index_parser.set_defaults(func=build_index)

    infer_parser = subparsers.add_parser(
        'infer',
        help='perform search on a given index'
    )
    infer_parser.add_argument(
        '--index_path',
        required=True,
        help='path to data index'
    )
    infer_parser.add_argument(
        '-n',
        type=int,
        help='number of nearest neighbors to fetch',
        default=10
    )
    infer_parser.add_argument(
        '--query_text',
        help='find similar items by query text'
    )
    infer_parser.add_argument(
        '--query_path',
        help='find similar items by item path'
    )
    infer_parser.add_argument(
        '--query_index_id',
        help='find simialr items by item in the index with a given id'
    )
    infer_parser.add_argument(
        '--data_reader_type',
        default='simple_dir',
        const='simple_dir',
        nargs='?',
        choices=('simple_dir',),
        help='index type (default: %(default)s)'
    )
    infer_parser.set_defaults(func=query)
    args = parser.parse_args()
    if not len(vars(args)):
        parser.print_help()
        raise SystemExit

    args.func(args)


if __name__ == '__main__':
    main()
