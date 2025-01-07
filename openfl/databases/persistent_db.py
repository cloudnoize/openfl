import json
import pickle
import sqlite3
import numpy as np
from threading import Lock
from typing import Dict, Iterator, Optional
import os
import logging

from openfl.utilities import LocalTensor, TensorKey, change_tags

logger = logging.getLogger(__name__)

__all__ = ['PersistentTensorDB']


class PersistentTensorDB:
    """
    The PersistentTensorDB class implements a database for storing tensors using SQLite.

    Attributes:
        conn: The SQLite connection object.
        cursor: The SQLite cursor object.
        lock: A threading Lock object used to ensure thread-safe operations.
    """

    def __init__(self, db_path: str = "") -> None:
        """Initializes a new instance of the PersistentTensorDB class."""
        full_path = "tensordb.sqlite"
        if db_path:
            full_path = os.path.join(db_path, full_path)
        logger.info("Initializing persistent db at %s",full_path)
        self.conn = sqlite3.connect(full_path, check_same_thread=False)
        self.cursor = self.conn.cursor()
        self.lock = Lock()
        self._create_model_tensors_table()
        self._create_task_results_table()
        self._create_key_value_store()

    def _create_model_tensors_table(self) -> None:
        """Create the database table for storing tensors if it does not exist."""
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS tensors (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                tensor_name TEXT NOT NULL,
                origin TEXT NOT NULL,
                round INTEGER NOT NULL,
                report INTEGER NOT NULL,
                tags TEXT,
                nparray BLOB NOT NULL
            )
        """)
        self.conn.commit()
    
    def _create_task_results_table(self) -> None:
        """Creates a table for storing task results."""
        create_table_query = """
        CREATE TABLE IF NOT EXISTS task_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            collaborator_name TEXT NOT NULL,
            round_number INTEGER NOT NULL,
            task_name TEXT NOT NULL,
            data_size INTEGER NOT NULL,
            named_tensors BLOB NOT NULL
        );
        """
        self.cursor.execute(create_table_query)
        self.conn.commit()
    
    def _create_key_value_store(self) -> None:
        """Create a key-value store table for storing additional metadata."""
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS key_value_store (
                key TEXT PRIMARY KEY,
                value REAL NOT NULL
            )
        """)
        self.conn.commit()

    def save_task_results(
        self,
        collaborator_name: str,
        round_number: int,
        task_name: str,
        data_size: int,
        named_tensors,
    ):
        """
        Saves task results to the task_results table.

        Args:
            collaborator_name (str): Collaborator name.
            round_number (int): Round number.
            task_name (str): Task name.
            data_size (int): Data size.
            named_tensors(List): list of binary representation of tensors.
        """
        serialized_blob = pickle.dumps(named_tensors)


        # Insert into the database
        insert_query = """
        INSERT INTO task_results 
        (collaborator_name, round_number, task_name, data_size, named_tensors)
        VALUES (?, ?, ?, ?, ?);
        """
        self.cursor.execute(
            insert_query,
            (collaborator_name, round_number, task_name, data_size, serialized_blob),
        )
        self.conn.commit()

    def get_task_result_by_id(self, task_result_id: int):
        """
        Retrieve a task result by its ID.

        Args:
            task_result_id (int): The ID of the task result to retrieve.

        Returns:
            A dictionary containing the task result details, or None if not found.
        """
        with self.lock:
            self.cursor.execute("""
                SELECT collaborator_name, round_number, task_name, data_size, named_tensors
                FROM task_results
                WHERE id = ?
            """, (task_result_id,))
            result = self.cursor.fetchone()
            if result:
                collaborator_name, round_number, task_name, data_size, serialized_blob = result
                serialized_tensors = pickle.loads(serialized_blob)
                return {
                    "collaborator_name": collaborator_name,
                    "round_number": round_number,
                    "task_name": task_name,
                    "data_size": data_size,
                    "named_tensors": serialized_tensors
                }
            return None

    def _serialize_array(self, array: np.ndarray) -> bytes:
        """Serialize a NumPy array into bytes for storing in SQLite."""
        return array.tobytes()

    def _deserialize_array(self, blob: bytes, dtype: Optional[np.dtype] = None) -> np.ndarray:
        """Deserialize bytes from SQLite into a NumPy array."""
        return np.frombuffer(blob, dtype=dtype)

    def __repr__(self) -> str:
        """Returns a string representation of the PersistentTensorDB."""
        with self.lock:
            self.cursor.execute("SELECT tensor_name, origin, round, report, tags FROM tensors")
            rows = self.cursor.fetchall()
            return f"PersistentTensorDB contents:\n{rows}"

    def finalize_round(self,tensor_key_dict: Dict[TensorKey, np.ndarray],round_number: int, best_score: float):
        with self.lock:
            try:
                # Begin transaction
                self.cursor.execute("BEGIN TRANSACTION")
                self._persist_tensors(tensor_key_dict)
                self._init_task_results_table()
                self._save_round_and_best_score(round_number,best_score)
                # Commit transaction
                self.conn.commit()
                logger.info(f"Committed model for round {round_number}, saved {len(tensor_key_dict)} model tensors with best_score {best_score}")
            except Exception as e:
                # Rollback transaction in case of an error
                self.conn.rollback()
                raise RuntimeError(f"Failed to finalize round: {e}")
            
    def _persist_tensors(self, tensor_key_dict: Dict[TensorKey, np.ndarray]) -> None:
        """Insert a dictionary of tensors into the SQLite database in a single transaction."""
        for tensor_key, nparray in tensor_key_dict.items():
            tensor_name, origin, fl_round, report, tags = tensor_key
            serialized_array = self._serialize_array(nparray)
            serialized_tags = json.dumps(tags) 
            self.cursor.execute("""
                INSERT INTO tensors (tensor_name, origin, round, report, tags, nparray)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (tensor_name, origin, fl_round, int(report), serialized_tags, serialized_array))
    
    def _init_task_results_table(self):
        """
        Creates a table for storing task results. Drops the table first if it already exists.
        """
        drop_table_query = "DROP TABLE IF EXISTS task_results"
        self.cursor.execute(drop_table_query)
        
        create_table_query = """
        CREATE TABLE IF NOT EXISTS task_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            collaborator_name TEXT NOT NULL,
            round_number INTEGER NOT NULL,
            task_name TEXT NOT NULL,
            data_size INTEGER NOT NULL,
            named_tensors BLOB NOT NULL
        );
        """
        self.cursor.execute(create_table_query)
    
    def _save_round_and_best_score(self, round_number: int, best_score: float) -> None:
        """Save the round number and best score as key-value pairs in the database."""
        # Create a table with key-value structure where values can be integer or float
        # Insert or update the round_number
        self.cursor.execute("""
            INSERT OR REPLACE INTO key_value_store (key, value)
            VALUES (?, ?)
        """, ("round_number", float(round_number)))

        # Insert or update the best_score
        self.cursor.execute("""
            INSERT OR REPLACE INTO key_value_store (key, value)
            VALUES (?, ?)
        """, ("best_score", float(best_score)))

    

    def load_tensors(self) -> Dict[TensorKey, np.ndarray]:
        """Load all tensors from the SQLite database and return them as a dictionary."""
        tensor_dict = {}
        with self.lock:
            self.cursor.execute("SELECT tensor_name, origin, round, report, tags, nparray FROM tensors")
            rows = self.cursor.fetchall()
            for row in rows:
                tensor_name, origin, fl_round, report, tags, nparray = row
                # Deserialize the JSON string back to a Python list
                deserialized_tags = tuple(json.loads(tags))
                tensor_key = TensorKey(tensor_name, origin, fl_round, report, deserialized_tags)
                tensor_dict[tensor_key] = self._deserialize_array(nparray)
        return tensor_dict

   
    def get_round_and_best_score(self) -> tuple[int, float]:
        """Retrieve the round number and best score from the database."""
        with self.lock:
            # Fetch the round_number
            self.cursor.execute("""
                SELECT value FROM key_value_store WHERE key = ?
            """, ("round_number",))
            round_number = self.cursor.fetchone()
            if round_number is None:
                round_number = -1
            else:
                round_number = int(round_number[0])  # Cast to int

            # Fetch the best_score
            self.cursor.execute("""
                SELECT value FROM key_value_store WHERE key = ?
            """, ("best_score",))
            best_score = self.cursor.fetchone()
            if best_score is None:
                best_score = 0
            else:
                best_score = float(best_score[0])  # Cast to float
        return round_number, best_score


    def clean_up(self, remove_older_than: int = 1) -> None:
        """Remove old entries from the database."""
        if remove_older_than < 0:
            return
        with self.lock:
            self.cursor.execute("SELECT MAX(round) FROM tensors")
            current_round = self.cursor.fetchone()[0]
            if current_round is None:
                return
            self.cursor.execute("""
                DELETE FROM tensors
                WHERE round <= ? AND report = 0
            """, (current_round - remove_older_than,))
            self.conn.commit()


    def close(self) -> None:
        """Close the SQLite database connection."""
        self.conn.close()

    def is_task_table_empty(self) -> bool:
        """Check if the task table is empty."""
        with self.lock:
            self.cursor.execute("SELECT COUNT(*) FROM task_results")
            count = self.cursor.fetchone()[0]
            return count == 0
