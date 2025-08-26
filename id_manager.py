import sqlite3
import uuid
from typing import List, Optional, Tuple
import os

from app.const import USERS_THREADS_DB_FILE_PATH


class IDManager:
    def __init__(self, db_path: str = USERS_THREADS_DB_FILE_PATH):
        """Initialize the ID Manager with SQLite database."""
        self.db_path = db_path
        self.init_database()

    def init_database(self):
        """Initialize the database with required tables only if database doesn't exist."""
        # Check if database file exists
        if not os.path.exists(self.db_path):
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                # Create users table
                cursor.execute(
                    """
                    CREATE TABLE users (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        user_id TEXT UNIQUE NOT NULL
                    )
                """
                )

                # Create threads table
                cursor.execute(
                    """
                    CREATE TABLE threads (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        thread_id TEXT NOT NULL,
                        user_id TEXT,
                        FOREIGN KEY (user_id) REFERENCES users (user_id),
                        UNIQUE(thread_id, user_id)
                    )
                """
                )

                # Create thread_history table
                cursor.execute(
                    """
                    CREATE TABLE thread_history (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        thread_id TEXT NOT NULL,
                        user_id TEXT NOT NULL,
                        user_query TEXT NOT NULL,
                        analyst_response TEXT NOT NULL,
                        entry_order INTEGER NOT NULL,
                        FOREIGN KEY (thread_id, user_id) REFERENCES threads (thread_id, user_id),
                        UNIQUE(thread_id, user_id, entry_order)
                    )
                """
                )

                conn.commit()
                print(f"Database created: {self.db_path}")

    def get_all_user_ids(self) -> List[str]:
        """
        Get all existing user IDs.
        Returns: List of user IDs
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT user_id FROM users ORDER BY user_id")
            return [row[0] for row in cursor.fetchall()]

    def create_user_id(self, user_id: str) -> bool:
        """
        Create a new user ID if it doesn't exist.
        Args:
            user_id: The user ID to create
        Returns:
            True if created successfully, False if already exists
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("INSERT INTO users (user_id) VALUES (?)", (user_id,))
                conn.commit()
                return True
        except sqlite3.IntegrityError:
            # User ID already exists
            return False

    def user_exists(self, user_id: str) -> bool:
        """Check if a user ID exists."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT 1 FROM users WHERE user_id = ?", (user_id,))
            return cursor.fetchone() is not None

    def get_all_thread_ids(
        self, user_id: Optional[str] = None
    ) -> List[Tuple[str, str]]:
        """
        Get all existing thread IDs.
        Args:
            user_id: Optional - filter threads by user_id
        Returns:
            List of tuples (thread_id, user_id)
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            if user_id:
                cursor.execute(
                    """
                    SELECT thread_id, user_id 
                    FROM threads 
                    WHERE user_id = ? 
                    ORDER BY thread_id
                """,
                    (user_id,),
                )
            else:
                cursor.execute(
                    """
                    SELECT thread_id, user_id 
                    FROM threads 
                    ORDER BY thread_id
                """
                )
            return cursor.fetchall()

    def create_thread_id(self, thread_id: str, user_id: str) -> bool:
        """
        Create a new thread ID if it doesn't exist for this user.
        Args:
            thread_id: The thread ID to create
            user_id: The user ID (required)
        Returns:
            True if created successfully, False if already exists for this user
        """
        if not user_id:
            return False

        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    INSERT INTO threads (thread_id, user_id) 
                    VALUES (?, ?)
                """,
                    (thread_id, user_id),
                )
                conn.commit()
                return True
        except sqlite3.IntegrityError:
            # Thread ID already exists for this user
            return False

    def thread_exists(self, thread_id: str, user_id: str) -> bool:
        """Check if a thread ID exists for a specific user."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT 1 FROM threads 
                WHERE thread_id = ? AND user_id = ?
            """,
                (thread_id, user_id),
            )
            return cursor.fetchone() is not None

    def generate_unique_user_id(self, prefix: str = "user") -> str:
        """Generate a unique user ID with optional prefix."""
        while True:
            unique_id = f"{prefix}_{uuid.uuid4().hex[:8]}"
            if not self.user_exists(unique_id):
                return unique_id

    def generate_unique_thread_id(self, user_id: str, prefix: str = "thread") -> str:
        """Generate a unique thread ID for a specific user."""
        while True:
            unique_id = f"{prefix}_{uuid.uuid4().hex[:8]}"
            if not self.thread_exists(unique_id, user_id):
                return unique_id

    def delete_user(self, user_id: str) -> bool:
        """
        Delete a user and all associated threads.
        Args:
            user_id: The user ID to delete
        Returns:
            True if deleted, False if user doesn't exist
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            # Check if user exists
            if not self.user_exists(user_id):
                return False

            # Delete associated threads first
            cursor.execute("DELETE FROM threads WHERE user_id = ?", (user_id,))

            # Delete user
            cursor.execute("DELETE FROM users WHERE user_id = ?", (user_id,))

            conn.commit()
            return True

    def delete_thread(self, thread_id: str, user_id: str) -> bool:
        """
        Delete a thread and all its history for a specific user.
        Args:
            thread_id: The thread ID to delete
            user_id: The user ID
        Returns:
            True if deleted, False if thread doesn't exist
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            # Delete thread history first
            cursor.execute(
                """
                DELETE FROM thread_history 
                WHERE thread_id = ? AND user_id = ?
            """,
                (thread_id, user_id),
            )

            # Delete thread
            cursor.execute(
                """
                DELETE FROM threads 
                WHERE thread_id = ? AND user_id = ?
            """,
                (thread_id, user_id),
            )
            deleted = cursor.rowcount > 0
            conn.commit()
            return deleted

    def get_thread_info(
        self, thread_id: str, user_id: str
    ) -> Optional[Tuple[str, str]]:
        """
        Get thread information for a specific user.
        Returns: (thread_id, user_id) or None if not found
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT thread_id, user_id 
                FROM threads 
                WHERE thread_id = ? AND user_id = ?
            """,
                (thread_id, user_id),
            )
            return cursor.fetchone()

    def get_user_info(self, user_id: str) -> Optional[str]:
        """
        Get user information.
        Returns: user_id or None if not found
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT user_id FROM users WHERE user_id = ?", (user_id,))
            result = cursor.fetchone()
            return result[0] if result else None

    def add_thread_entry(
        self, thread_id: str, user_id: str, user_query: str, analyst_response: str
    ) -> bool:
        """
        Add a new entry to thread history.
        Args:
            thread_id: The thread ID
            user_id: The user ID
            user_query: The user's query
            analyst_response: The analyst's response
        Returns:
            True if added successfully, False if thread doesn't exist
        """
        if not self.thread_exists(thread_id, user_id):
            return False

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            # Get the next order number for this thread and user
            cursor.execute(
                """
                SELECT COALESCE(MAX(entry_order), 0) + 1 
                FROM thread_history 
                WHERE thread_id = ? AND user_id = ?
            """,
                (thread_id, user_id),
            )
            next_order = cursor.fetchone()[0]

            # Insert the new entry
            cursor.execute(
                """
                INSERT INTO thread_history (thread_id, user_id, user_query, analyst_response, entry_order)
                VALUES (?, ?, ?, ?, ?)
            """,
                (thread_id, user_id, user_query, analyst_response, next_order),
            )

            conn.commit()
            return True

    def get_thread_history(
        self, thread_id: str, user_id: str
    ) -> List[Tuple[str, str, int]]:
        """
        Get all history entries for a thread and user, ordered by entry_order.
        Args:
            thread_id: The thread ID
            user_id: The user ID
        Returns:
            List of tuples (user_query, analyst_response, entry_order)
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT user_query, analyst_response, entry_order
                FROM thread_history
                WHERE thread_id = ? AND user_id = ?
                ORDER BY entry_order ASC
            """,
                (thread_id, user_id),
            )
            return cursor.fetchall()

    def get_thread_entry_count(self, thread_id: str, user_id: str) -> int:
        """
        Get the number of entries in a thread's history for a specific user.
        Args:
            thread_id: The thread ID
            user_id: The user ID
        Returns:
            Number of entries
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT COUNT(*) FROM thread_history 
                WHERE thread_id = ? AND user_id = ?
            """,
                (thread_id, user_id),
            )
            return cursor.fetchone()[0]

    def clear_thread_history(self, thread_id: str, user_id: str) -> bool:
        """
        Clear all history entries for a thread and user.
        Args:
            thread_id: The thread ID
            user_id: The user ID
        Returns:
            True if cleared, False if thread doesn't exist
        """
        if not self.thread_exists(thread_id, user_id):
            return False

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                DELETE FROM thread_history 
                WHERE thread_id = ? AND user_id = ?
            """,
                (thread_id, user_id),
            )
            conn.commit()
            return True
