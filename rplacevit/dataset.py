import sqlite3
import numpy as np
import os
import math
import re
import pickle
from torch.utils.data import Dataset
from collections import defaultdict

class RPlaceDataset(Dataset):
    """
    Base class for rplace-igpt datasets
    Manages connections to SQLite partitions, end states, and user features
    """
    def __init__(self, partitions_dir="partitions", end_states_dir="end_states", 
                 view_width=64, palette_size=32, user_class_size=8,
                 single_partition=None, partition_size=1000000, use_user_features=False, user_features_db=None,
                 min_x=0, min_y=0, max_x=2048, max_y=2048):
        """
        Initialize the dataset
        
        Args:
            partitions_dir (str): directory containing SQLite partitions
            end_states_dir (str): directory containing end states
            view_width (int): width and height of the view
            palette_size (int): number of colors in the palette
            user_class_size (int): number of user classes
            single_partition (int): index of a single partition to use, or None to use all partitions, for testing / overfit purposes
            partition_size (int): number of rows in each partition
            use_user_features (bool): whether to use user features
            user_features_db (str): path to the SQLite database file containing user features
            min_x (int): minimum x-coordinate of the region to use
            min_y (int): minimum y-coordinate of the region to use
            max_x (int): maximum x-coordinate of the region to use
            max_y (int): maximum y-coordinate of the region to use
        """
        self.partitions_dir = partitions_dir
        self.end_states_dir = end_states_dir
        self.view_width = view_width
        self.palette_size = palette_size
        self.user_class_size = user_class_size
        self.single_partition = single_partition
        self.partition_size = partition_size
        self.use_user_features = use_user_features
        self.user_features_db = user_features_db
        self.min_x = min_x
        self.min_y = min_y
        self.max_x = max_x
        self.max_y = max_y
        self.connections = {}
        self.total_rows, self.id_map = self._get_total_rows_and_id_map()
        self.end_states = []
        print("Dataset initialized")
        
    def _get_total_rows_and_id_map(self):
        total = 0
        id_map = {}
        
        if self.single_partition is not None:
            conn = self._get_connection(self.single_partition)
            cursor = conn.cursor()
            cursor.execute("""
                SELECT id FROM pixels 
                WHERE x >= ? AND x < ? AND y >= ? AND y < ?
                ORDER BY id
            """, (self.min_x, self.max_x, self.min_y, self.max_y))
            for row in cursor.fetchall():
                id_map[total] = row[0]
                total += 1
        else:
            total_partitions = len([f for f in os.listdir(self.partitions_dir) if f.startswith("partition_") and f.endswith(".db")])
            for partition in range(total_partitions):
                conn = self._get_connection(partition)
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT id FROM pixels 
                    WHERE x >= ? AND x < ? AND y >= ? AND y < ?
                    ORDER BY id
                """, (self.min_x, self.max_x, self.min_y, self.max_y))
                for row in cursor.fetchall():
                    id_map[total] = (partition, row[0])
                    total += 1
        
        print(f"Total rows in selected region: {total}")
        return total, id_map

    def _get_total_rows(self):
        """
        Get the total number of rows in the dataset
        
        Returns:
            int: total number of rows
        """
       
        if self.single_partition is not None:
            conn = self._get_connection(self.single_partition)
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM pixels")
            total = cursor.fetchone()[0]
        else:
            total_partitions = len([f for f in os.listdir(self.partitions_dir) if f.startswith("partition_") and f.endswith(".db")])
            last_partition = total_partitions - 1
            conn = self._get_connection(last_partition)
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM pixels")
            total = cursor.fetchone()[0]
            total += (total_partitions - 1) * self.partition_size
            
        print(f"Total rows: {total}")
        return total

    def _get_connection(self, partition: int) -> sqlite3.Connection:
        """
        Get a connection to the SQLite database for a partition
        
        Args:
            partition (int): partition number
            
        Returns:
            sqlite3.Connection: connection to the database
        """
        if partition not in self.connections:
            db_path = os.path.join(self.partitions_dir, f'partition_{partition}.db')
            self.connections[partition] = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
            self.connections[partition].execute("PRAGMA query_only = ON")
            self.connections[partition].execute("PRAGMA cache_size = 100000")
            self.connections[partition].execute("PRAGMA mmap_size = 10000000000")
            self.connections[partition].execute("PRAGMA journal_mode = OFF")
            self.connections[partition].execute("PRAGMA synchronous = OFF")
            self.connections[partition].execute("PRAGMA temp_store = MEMORY")
            self.connections[partition].row_factory = sqlite3.Row
        return self.connections[partition]
    
    def load_end_states(self):
        """
        Load the end states from the end_states_dir
        """
        os.makedirs(self.end_states_dir, exist_ok=True)
        
        if self.single_partition is not None:
            end_state_file = os.path.join(self.end_states_dir, f"end_state_{self.single_partition}.txt")
            end_state = np.loadtxt(end_state_file, delimiter=";")
            print(f"Loaded end state from {end_state_file}")
            self.end_states = [end_state]
            return
                
        end_states = []
        end_state_files = sorted(
            [f for f in os.listdir(self.end_states_dir) if f.endswith(".txt")],
            key=lambda x: [int(t) if t.isdigit() else t.lower() for t in re.split('([0-9]+)', x)]
        )
        
        for end_state_file in end_state_files:
            end_state = np.loadtxt(os.path.join(self.end_states_dir, end_state_file), delimiter=";")
            end_states.append(end_state)
            print(f"Loaded end state from {end_state_file}")
        
        invalid_files = [f for f in os.listdir(self.end_states_dir) if not f.endswith(".txt")]
        for invalid_file in invalid_files:
            print(f"Invalid file in end_states_dir: {invalid_file}")
            
        self.end_states = end_states
    
    def compute_users_features(self, force: bool = False, chunk_size: int = 100000) -> dict:
        """
        Compute features for all users in the dataset using a chunk-based approach
        
        Args:
            force (bool): force re-computation of features
            chunk_size (int): number of rows to process in each chunk
        
        Returns:
            dict: user features
        """
        if os.path.exists('user_features.pkl') and not force:
            print("Loading existing user features...")
            with open('user_features.pkl', 'rb') as f:
                self.user_features = pickle.load(f)
            self.user_features_size = next(iter(self.user_features.values())).shape[0]
            return self.user_features

        print("Computing user features...")

        user_data = defaultdict(lambda: {
            'x_sum': 0, 'y_sum': 0,
            'x_sq_sum': 0, 'y_sq_sum': 0,
            'timestamp_sum': 0,
            'count': 0,
            'colors': np.zeros(32, dtype=int)
        })

        total_rows = self._get_total_rows()
        processed_rows = 0

        min_timestamp = float('inf')
        max_timestamp = float('-inf')

        for partition in range(len([f for f in os.listdir(self.partitions_dir) if f.startswith("partition_") and f.endswith(".db")])):
            conn = self._get_connection(partition)
            cursor = conn.cursor()

            offset = 0
            while True:
                cursor.execute('''
                    SELECT user_id, color_id, x, y, timestamp
                    FROM pixels
                    LIMIT ? OFFSET ?
                ''', (chunk_size, offset))
                
                chunk = cursor.fetchall()
                if not chunk:
                    break

                for row in chunk:
                    user_id, color_id, x, y, timestamp = row
                    user = user_data[user_id]
                    user['x_sum'] += x
                    user['y_sum'] += y
                    user['x_sq_sum'] += x * x
                    user['y_sq_sum'] += y * y
                    user['timestamp_sum'] += timestamp
                    user['count'] += 1
                    user['colors'][color_id] += 1
                    min_timestamp = min(min_timestamp, timestamp)
                    max_timestamp = max(max_timestamp, timestamp)

                offset += chunk_size
                processed_rows += len(chunk)
                print(f"Processed {processed_rows}/{total_rows} rows ({processed_rows/total_rows*100:.2f}%)")

        print("Computing final user features...")
        
        user_ids = np.array(list(user_data.keys()))
        x_sums = np.array([data['x_sum'] for data in user_data.values()])
        y_sums = np.array([data['y_sum'] for data in user_data.values()])
        x_sq_sums = np.array([data['x_sq_sum'] for data in user_data.values()])
        y_sq_sums = np.array([data['y_sq_sum'] for data in user_data.values()])
        timestamp_sums = np.array([data['timestamp_sum'] for data in user_data.values()])
        counts = np.array([data['count'] for data in user_data.values()])
        colors = np.array([data['colors'] for data in user_data.values()])

        x_means = x_sums / counts
        y_means = y_sums / counts
        x_stds = np.sqrt(x_sq_sums / counts - x_means ** 2)
        y_stds = np.sqrt(y_sq_sums / counts - y_means ** 2)
        avg_timestamps = timestamp_sums / counts
        color_hists = colors / counts[:, np.newaxis]

        x_means_norm = x_means / 1000
        y_means_norm = y_means / 1000
        x_stds_norm = x_stds / 1000
        y_stds_norm = y_stds / 1000
        counts_norm = np.minimum(counts / 1000, 1.0)
        timestamp_norm = (avg_timestamps - min_timestamp) / (max_timestamp - min_timestamp)

        features = np.column_stack([
            x_means_norm, y_means_norm, x_stds_norm, y_stds_norm,
            counts_norm, timestamp_norm, color_hists
        ])

        self.user_features = dict(zip(user_ids, features))
        self.user_features_size = features.shape[1]

        with open('user_features.pkl', 'wb') as f:
            pickle.dump(self.user_features, f)

        print(f"Computed features for {len(self.user_features)} users")
        return self.user_features
            
    def store_user_features(self, db_path='user_features.db'):
        """
        Store user features in an SQLite database with user ID as the primary key.
        
        Args:
            db_path (str): Path to the SQLite database file
        """
        print(f"Storing user features in {db_path}...")

        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        cursor.execute('''
        CREATE TABLE IF NOT EXISTS user_features (
            user_id INTEGER PRIMARY KEY,
            features BLOB
        )
        ''')

        data = []
        for user_id, features in self.user_features.items():
            try:
                int_user_id = int(user_id)
                serialized_features = pickle.dumps(features)
                data.append((int_user_id, sqlite3.Binary(serialized_features)))
            except ValueError:
                print(f"Warning: Skipping invalid user_id: {user_id}")

        # Use executemany for efficient bulk insert
        cursor.executemany('INSERT OR REPLACE INTO user_features (user_id, features) VALUES (?, ?)', data)

        conn.commit()
        conn.close()

        print(f"Stored features for {len(data)} users in {db_path}")

        self.user_features_db = db_path      
        
    def get_user_features(self, user_id):
        """
        Retrieve user features for a specific user ID from the SQLite database.
        
        Args:
            user_id (int): The ID of the user whose features to retrieve
        
        Returns:
            np.array: User features for the specified user ID, or None if not found
        """
        if not hasattr(self, 'user_features_db'):
            raise ValueError("User features database not set. Call store_user_features() first.")

        if not hasattr(self, 'user_features_conn'):
            self.user_features_conn = sqlite3.connect(f"file:{self.user_features_db}?mode=ro", uri=True)
            self.user_features_conn.execute("PRAGMA query_only = ON")

        cursor = self.user_features_conn.cursor()
        cursor.execute('SELECT features FROM user_features WHERE user_id = ?', (int(user_id),))
        result = cursor.fetchone()

        if result:
            return pickle.loads(result[0])
        else:
            return None
    
    def _create_view(self, x, y, timestamp, user_features):
        """
        Helper method to create the view input for a given pixel
        
        Args:
            x (int): x-coordinate of the center pixel
            y (int): y-coordinate of the center pixel
            timestamp (int): timestamp of the pixel placement
            user_features (np.array): features of the user who placed the pixel
            
        Returns:
            np.array: view input
        """
        view_x_min = max(x - self.view_width // 2, self.min_x)
        view_x_max = min(x + self.view_width // 2, self.max_x)
        view_y_min = max(y - self.view_width // 2, self.min_y)
        view_y_max = min(y + self.view_width // 2, self.max_y)
        
        num_user_features = len(user_features)
        
        view = np.zeros((self.view_width, self.view_width, self.palette_size + num_user_features), dtype=np.float32)
        partition = self.single_partition if self.single_partition is not None else math.floor(timestamp / self.partition_size)
        if partition == 0:
            source_view = np.zeros((self.view_width, self.view_width), dtype=np.uint8)
        else:
            partition = min(partition, len(self.end_states) - 1)
            source_view = self.end_states[partition - 1][view_y_min:view_y_max, view_x_min:view_x_max]

        source_view_int = source_view.astype(int)
        view[:, :, source_view_int] = 1
        
        for i, feature in enumerate(user_features):
            view[:, :, self.palette_size + i] = feature
        
        conn = self._get_connection(partition)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT x, y, color_id, MAX(timestamp) as timestamp
            FROM pixels
            WHERE timestamp < ? AND x >= ? AND x < ? AND y >= ? AND y < ?
            GROUP BY x, y
        ''', (timestamp, view_x_min, view_x_max, view_y_min, view_y_max))
        
        for pixel in cursor.fetchall():
            px, py, pcolor_id, ptimestamp = pixel['x'], pixel['y'], pixel['color_id'], pixel['timestamp']
            view_x = px - view_x_min
            view_y = py - view_y_min
            view[view_y, view_x, pcolor_id] = 1
            
        return view.transpose(2, 0, 1)

    def __len__(self) -> int:
        """
        Get the total number of rows in the dataset
        
        Returns:
            int: total number of rows
        """
        return self.total_rows

    def __getitem__(self, idx) -> tuple:
        """
        Get an item from the dataset
        
        Args:
            idx (int): index of the item
        """
        raise NotImplementedError

    def __del__(self):
        """
        Close all connections when the object is deleted
        """
        for conn in self.connections.values():
            conn.close()
            
class RPlaceColorDataset(RPlaceDataset):
    """
    Dataset for predicting the next color of the center pixel in a view
    """
    def __getitem__(self, idx) -> tuple:
        """
        Get an item from the dataset
        target is the one-hot encoded most probable color of the pixel in the next minute
        
        Args:
            idx (int): index of the item
            
        Returns:
            tuple: (view, target)
        """
        if idx >= len(self.id_map):
            raise IndexError("Index out of range")
        
        real_id = self.id_map[idx]
        
        if self.single_partition is not None:
            partition = self.single_partition
            pixel_id = real_id
        else:
            partition, pixel_id = real_id
        
        conn = self._get_connection(partition)
        cursor = conn.cursor()
        
        cursor.execute('SELECT x, y, color_id, user_id, timestamp FROM pixels WHERE id = ?', (pixel_id,))
        row = cursor.fetchone()
        x, y, color_id, user_id, timestamp = row['x'], row['y'], row['color_id'], row['user_id'], row['timestamp']
        
        user_features = np.zeros(0)
        if self.use_user_features:
            user_features = self.get_user_features(user_id)
            if user_features is None:
                user_features = np.zeros(self.user_features_size)
                
        view = self._create_view(x, y, timestamp, user_features)
        
        one_minute_later = timestamp + 60
        cursor.execute('''
            SELECT color_id, COUNT(*) as count
            FROM pixels
            WHERE x = ? AND y = ? AND timestamp > ? AND timestamp <= ?
            GROUP BY color_id
            ORDER BY count DESC
            LIMIT 1
        ''', (x, y, timestamp, one_minute_later))
        
        most_probable_color = cursor.fetchone()
        
        if most_probable_color is None:
            most_probable_color_id = color_id
        else:
            most_probable_color_id = most_probable_color['color_id']
        
        target = np.zeros(self.palette_size, dtype=np.uint8)
        target[most_probable_color_id] = 1
        
        target = target.astype(np.float32)
            
        return view, target

class RPlaceTimeDataset(RPlaceDataset):
    """
    Dataset for predicting the time before the next change of the center pixel in a view
    Allow to predict the next pixel to change before using the color transformer
    """
    def __getitem__(self, idx) -> tuple:
        """
        Get an item from the dataset
        target is 16x16, with the normalized time before the next change for each pixel in the center view

        Args:
            idx (int): index of the item
            
        Returns:
            tuple: (view, target)
        """
        idx = idx + 1
        partition = self.single_partition if self.single_partition is not None else math.floor(idx / self.partition_size)
        if self.single_partition is not None:
            idx = idx % self.partition_size
            idx += self.partition_size * partition

        conn = self._get_connection(partition)
        cursor = conn.cursor()

        cursor.execute('SELECT x, y, color_id, user_id, timestamp FROM pixels WHERE id = ?', (idx,))
        initial_pixel = cursor.fetchone()
        x, y, color_id, user_id, timestamp = initial_pixel['x'], initial_pixel['y'], initial_pixel['color_id'], initial_pixel['user_id'], initial_pixel['timestamp']
            
        user_features = np.zeros(0)
        if self.use_user_features:
            user_features = self.get_user_features(user_id)
            if user_features is None:
                user_features = np.zeros(self.user_features_size)
                
        view = self._create_view(x, y, timestamp, user_features)
        
        center_x_min = x - 8
        center_x_max = x + 8
        center_y_min = y - 8
        center_y_max = y + 8

        cursor.execute('''
            SELECT x, y, MIN(timestamp) as next_timestamp
            FROM pixels
            WHERE timestamp > ? AND x >= ? AND x < ? AND y >= ? AND y < ?
            GROUP BY x, y
        ''', (timestamp, center_x_min, center_x_max, center_y_min, center_y_max))
        
        target = np.ones((16, 16), dtype=np.float32)

        for pixel in cursor.fetchall():
            px, py, next_timestamp = pixel['x'], pixel['y'], pixel['next_timestamp']
            target_x = px - center_x_min
            target_y = py - center_y_min
            time_diff = next_timestamp - timestamp
            normalized_time = min(time_diff / 3600, 1.0)
            target[target_y, target_x] = normalized_time

        return view, target
            
if __name__ == "__main__":
    dataset = RPlaceColorDataset(single_partition=0, min_x=0, min_y=0, max_x=256, max_y=256)
    dataset.load_end_states()
    print(len(dataset))
    print(dataset[0])
    
    #dataset = RPlaceTimeDataset(single_partition=0)
    #dataset.load_end_states()
    #print(len(dataset))
    #print(dataset[0])