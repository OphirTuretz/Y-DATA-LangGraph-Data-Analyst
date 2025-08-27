import os

# List of files to delete
files_to_delete = [
    "users_threads.db",
    "graph_state_store.db",
    "graph_state_checkpointer.db",
]

for filename in files_to_delete:
    try:
        if os.path.exists(filename):
            os.remove(filename)
            print(f"Deleted: {filename}")
        else:
            print(f"Not found: {filename}")
    except Exception as e:
        print(f"Error deleting {filename}: {e}")
