# -*- coding:utf-8 -*-
import numpy as np
import h5py
import os
import shutil
import sqlite3
import time


class FlatIndex(object):

    def __init__(self):
        self.connected = False

    def create(
        self,
        path_prefix,
        vector_size=512,
        bucket_size=256,
        force=False
    ):

        metadb_path = os.path.join(path_prefix, 'meta.db')

        if os.path.exists(path_prefix):
            if os.path.isfile(metadb_path):
                if not force:
                    raise FileExistsError(
                        (
                            'Index at `{0}` already exists.\n'
                            'Use FlatIndex.connect(\'{0}\') to connect to it, '
                            'or FlatIndex.create(\'{0}\', force=True) '
                            'in order to overwrite it.'
                        ).format(path_prefix)
                    )
                else:
                    shutil.rmtree(path_prefix)
                    os.makedirs(path_prefix)
        else:
            os.makedirs(path_prefix)

        self.path_prefix = path_prefix
        self.vector_size = vector_size
        self.bucket_size = bucket_size

        self.metadb = sqlite3.connect(metadb_path)
        cursor = self.metadb.cursor()

        cursor.execute(
            '''
            CREATE TABLE IF NOT EXISTS config (
                id integer primary key CHECK (id = 0),
                vector_size integer,
                bucket_size integer
            )
            '''
        )

        try:
            cursor.execute(
                '''
                INSERT into
                    config(id, vector_size, bucket_size)
                VALUES(0, ?, ?)
                ''',
                (self.vector_size, self.bucket_size)
            )
        except sqlite3.IntegrityError:
            pass

        cursor.execute(
            '''
            CREATE TABLE IF NOT EXISTS metadata (
                id integer primary key autoincrement,
                bucket_id integer,
                bucket_item_id integer,
                local_path text,
                UNIQUE(local_path)
            )
            '''
        )

        cursor.execute(
            '''
            CREATE TABLE IF NOT EXISTS current_bucket (
                id integer primary key CHECK (id = 0),
                bucket_id integer,
                n_items integer
            )
            '''
        )

        try:
            cursor.execute(
                '''
                INSERT into
                    current_bucket(id, bucket_id, n_items)
                VALUES(0, 0, 0)
                '''
            )
        except sqlite3.IntegrityError:
            pass

        self.metadb.commit()
        self.connected = True

    def connect(self, path_prefix):
        metadb_path = os.path.join(path_prefix, 'meta.db')

        if not os.path.exists(path_prefix) or not os.path.isfile(metadb_path):
            raise FileNotFoundError(
                'No index meta db found at `{0}` '
                'Use FlatIndex.create(\'{0}\') in order to '
                'initalize new index.'.format(metadb_path)
            )

        self.path_prefix = path_prefix
        self.metadb = sqlite3.connect(metadb_path)
        cursor = self.metadb.cursor()
        self.vector_size, self.bucket_size = cursor.execute(
                '''
                SELECT
                    vector_size,
                    bucket_size
                FROM
                    config
                WHERE
                    id = 0
                ''',
            ).fetchall()[0]
        self.connected = True

    def check_connection(self):
        if not self.connected:
            raise ConnectionError(  # TODO connection error??
                'Not connected to any index directory. '
                'Please create index first, or connect to the existing one.'
            )
        return True

    def get_current_bucket_id(self):
        self.check_connection()
        cursor = self.metadb.cursor()
        current_bucket = cursor.execute(
            '''
            SELECT
                bucket_id
            FROM
                current_bucket
            WHERE
                id = 0
            '''
        ).fetchall()
        self.metadb.commit()
        return current_bucket[0][0]

    def increment_current_bucket(self):
        self.check_connection()
        cursor = self.metadb.cursor()
        cursor.execute(
            '''
            UPDATE
                current_bucket
            SET
                bucket_id = CASE WHEN n_items = ? THEN (bucket_id + 1) ELSE bucket_id END,
                n_items = CASE WHEN n_items = ? THEN 0 ELSE (n_items + 1) END
            WHERE
                id = 0
            ''',
            (self.bucket_size, self.bucket_size)
        )
        self.metadb.commit()

    def add_item(self, path, vec):
        self.check_connection()
        # TODO: probably merge this with INSERT request
        cursor = self.metadb.cursor()
        path_exists = cursor.execute(
            '''
            SELECT
                EXISTS(
                    SELECT
                        1
                    FROM
                        metadata
                    WHERE
                        local_path = ?
                )
            ''',
            (path,)
        ).fetchall()[0][0]
        self.metadb.commit()
        if path_exists:
            return (-1, -1, -1)

        bucket_id = self.get_current_bucket_id()

        bucket_path = os.path.join(
            self.path_prefix,
            '{}.h5'.format(bucket_id)
        )

        if os.path.exists(bucket_path):
            with h5py.File(bucket_path, 'a') as f:
                data = f['v']
                data.resize((data.shape[0] + 1,) + vec.shape)
                data[data.shape[0] - 1] = vec
                bucket_item_id = data.shape[0] - 1
                f.flush()
        else:
            with h5py.File(bucket_path, 'w') as f:
                f.create_dataset(
                    'v',
                    data=np.expand_dims(vec, axis=0),
                    dtype=vec.dtype,
                    chunks=True,
                    maxshape=(None,) + vec.shape
                )
                bucket_item_id = 0

        cursor = self.metadb.cursor()
        cursor.execute(
            '''
            INSERT into
                metadata(bucket_id, bucket_item_id, local_path)
            VALUES(?, ?, ?)
            ''',
            (bucket_id, bucket_item_id, path)
        )
        item_id = cursor.lastrowid
        self.metadb.commit()
        self.increment_current_bucket()

        return item_id, bucket_id, bucket_item_id

    def get_item_vector(self, path=None, id=None):
        self.check_connection()
        cursor = self.metadb.cursor()
        if id:
            bucket_id, bucket_item_id = cursor.execute(
                '''
                SELECT
                    bucket_id,
                    bucket_item_id
                FROM
                    metadata
                WHERE
                    id = ?
                ''',
                (id,)
            ).fetchall()[0]
        else:
            bucket_id, bucket_item_id = cursor.execute(
                '''
                SELECT
                    bucket_id,
                    bucket_item_id
                FROM
                    metadata
                WHERE
                    local_path = ?
                ''',
                path
            ).fetchall()[0]
        self.metadb.commit()

        bucket_path = os.path.join(
            self.path_prefix,
            '{}.h5'.format(bucket_id)
        )

        with h5py.File(bucket_path, 'r') as hf:
            vec = hf['v'][bucket_item_id]

        return vec

    def get_item_path(self, bucket_id=None, bucket_item_id=None):
        self.check_connection()
        cursor = self.metadb.cursor()
        path = cursor.execute(
            '''
            SELECT
                local_path
            FROM
                metadata
            WHERE
                bucket_id = ?
                AND bucket_item_id = ?
            ''',
            (bucket_id, bucket_item_id)
        ).fetchall()[0][0]

        self.metadb.commit()
        return path

    def get_nn_by_vec(self, vec, n=10):
        self.check_connection()
        current_bucket_id = self.get_current_bucket_id()
        if current_bucket_id == 0:
            buckets = [0]
        else:
            buckets = range(0, current_bucket_id)
        sims = []
        for bucket_id in buckets:
            bucket_path = os.path.join(
                self.path_prefix,
                '{}.h5'.format(bucket_id)
            )

            with h5py.File(bucket_path, 'r') as hf:
                bucket = hf['v'][:]
            bucket_sims = np.dot(vec, bucket.T)

            for bucket_item_id in range(bucket_sims.shape[0]):
                sims.append(
                    (bucket_id, bucket_item_id, bucket_sims[bucket_item_id])
                )
        sims = sorted(sims, reverse=True, key=lambda x: x[2])[:n]
        sims = [(self.get_item_path(s[0], s[1]), s[2]) for s in sims]
        return sims
