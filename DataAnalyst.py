import streamlit as st
from datetime import datetime
from typing import List, Tuple

from id_manager import IDManager
from app.const import DATE_TIME_PATTERN
from engine import process_user_query
from data import Dataset


# Page config
st.set_page_config(page_title="Data Analyst Agent", layout="centered")


# Initialize ID Manager
@st.cache_resource
def get_id_manager():
    return IDManager()


id_manager = get_id_manager()


# History management functions
def load_thread_history(user_id: str, thread_id: str) -> List[Tuple[str, str, int]]:
    """Load conversation history for a specific thread and user."""
    try:
        history = id_manager.get_thread_history(thread_id, user_id)
        return history
    except Exception as e:
        st.error(f"Error loading thread history: {str(e)}")
        return []


def display_conversation_history(history: List[Tuple[str, str, int]]):
    """Display conversation history in Streamlit."""
    if not history:
        return

    st.markdown("### 📜 Conversation History")

    # Sort by entry_order to ensure chronological display
    sorted_history = sorted(history, key=lambda x: x[2])

    for user_query, analyst_response, entry_order in sorted_history:
        # User message
        with st.container():
            st.markdown("**👤 You:**")
            st.markdown(f"{user_query}")

        # Agent response
        with st.container():
            st.markdown("**🤖 Agent:**")
            st.markdown(f"{analyst_response}")

        st.markdown("---")


def save_conversation_entry(
    user_id: str, thread_id: str, user_query: str, analyst_response: str
) -> bool:
    """Save a new conversation entry to the database."""
    try:
        success = id_manager.add_thread_entry(
            thread_id, user_id, user_query, analyst_response
        )
        if success:
            log(
                f"Conversation entry saved to DB for user {user_id}, thread {thread_id}"
            )
        else:
            st.error(f"Failed to save conversation entry to database")
        return success
    except Exception as e:
        st.error(f"Error saving conversation entry: {str(e)}")
        return False


def check_thread_has_history(user_id: str, thread_id: str) -> bool:
    """Check if a thread has existing conversation history."""
    try:
        entry_count = id_manager.get_thread_entry_count(thread_id, user_id)
        return entry_count > 0
    except Exception as e:
        st.error(f"Error checking thread history: {str(e)}")
        return False


# Initialize session state variables
if "current_user_id" not in st.session_state:
    st.session_state.current_user_id = None

if "current_thread_id" not in st.session_state:
    st.session_state.current_thread_id = None

if "ids_configured" not in st.session_state:
    st.session_state.ids_configured = False

if "response" not in st.session_state:
    st.session_state.response = ""

if "submitted" not in st.session_state:
    st.session_state.submitted = False

if "logs" not in st.session_state:
    st.session_state.logs = []

if "conversation_history" not in st.session_state:
    st.session_state.conversation_history = []

if "thread_has_history" not in st.session_state:
    st.session_state.thread_has_history = False


def log(message, print_to_sidebar=True):
    message = f"[{datetime.now().strftime(DATE_TIME_PATTERN)}] {message}"
    st.session_state.logs.append(message)
    print(message)

    if st.session_state.get("developer_mode", False) and print_to_sidebar:
        st.sidebar.text(message)


# Cache dataset loading
@st.cache_resource(show_spinner="Loading dataset, please wait...")
def load_bitext_dataset():
    ds = Dataset()
    return ds


# Title
st.title("🤖 Data Analyst Agent")

# Upper Bar - User and Thread Management
st.markdown("---")
with st.container():
    st.markdown("### 👤 User & Thread Management")

    col1, col2, col3 = st.columns([2, 2, 1])

    with col1:
        st.markdown("**User Management**")

        # Get all users
        all_users = id_manager.get_all_user_ids()
        user_options = ["Select User"] + all_users + ["Create New User"]

        selected_user_option = st.selectbox(
            "Choose User:",
            user_options,
            index=(
                0
                if not st.session_state.current_user_id
                else (
                    user_options.index(st.session_state.current_user_id)
                    if st.session_state.current_user_id in user_options
                    else 0
                )
            ),
            key="user_selectbox",
        )

        # Handle user selection
        if selected_user_option == "Create New User":
            new_user_id = st.text_input(
                "Enter new User ID:", key="new_user_input", placeholder="e.g., john_doe"
            )

            if st.button("Create User", key="create_user_btn"):
                if new_user_id.strip():
                    if id_manager.create_user_id(new_user_id.strip()):
                        st.session_state.current_user_id = new_user_id.strip()
                        st.session_state.current_thread_id = None
                        st.session_state.conversation_history = []
                        st.session_state.thread_has_history = False
                        st.success(f"✅ User '{new_user_id}' created successfully!")
                        log(f"Created new user: {new_user_id}")
                        st.rerun()
                    else:
                        st.error(f"❌ User '{new_user_id}' already exists!")
                else:
                    st.error("❌ Please enter a valid User ID")

        elif selected_user_option != "Select User":
            if st.session_state.current_user_id != selected_user_option:
                st.session_state.current_user_id = selected_user_option
                st.session_state.current_thread_id = None
                st.session_state.conversation_history = []
                st.session_state.thread_has_history = False
                log(f"Selected user: {selected_user_option}")

    with col2:
        st.markdown("**Thread Management**")

        if st.session_state.current_user_id:
            # Get threads for current user
            user_threads = id_manager.get_all_thread_ids(
                st.session_state.current_user_id
            )
            thread_options = (
                ["Select Thread"] + [t[0] for t in user_threads] + ["Create New Thread"]
            )

            selected_thread_option = st.selectbox(
                "Choose Thread:",
                thread_options,
                index=(
                    0
                    if not st.session_state.current_thread_id
                    else (
                        thread_options.index(st.session_state.current_thread_id)
                        if st.session_state.current_thread_id in thread_options
                        else 0
                    )
                ),
                key="thread_selectbox",
            )

            # Handle thread selection
            if selected_thread_option == "Create New Thread":
                new_thread_id = st.text_input(
                    "Enter new Thread ID:",
                    key="new_thread_input",
                    placeholder="e.g., conversation_001",
                )

                if st.button("Create Thread", key="create_thread_btn"):
                    if new_thread_id.strip():
                        if id_manager.create_thread_id(
                            new_thread_id.strip(), st.session_state.current_user_id
                        ):
                            st.session_state.current_thread_id = new_thread_id.strip()
                            st.session_state.conversation_history = []
                            st.session_state.thread_has_history = False
                            st.success(
                                f"✅ Thread '{new_thread_id}' created successfully!"
                            )
                            log(
                                f"Created new thread: {new_thread_id} for user: {st.session_state.current_user_id}"
                            )
                            st.rerun()
                        else:
                            st.error(f"❌ Thread '{new_thread_id}' already exists!")
                    else:
                        st.error("❌ Please enter a valid Thread ID")

            elif selected_thread_option != "Select Thread":
                if st.session_state.current_thread_id != selected_thread_option:
                    st.session_state.current_thread_id = selected_thread_option
                    log(f"Selected thread: {selected_thread_option}")

                    # Load conversation history for existing thread
                    st.session_state.conversation_history = load_thread_history(
                        st.session_state.current_user_id,
                        st.session_state.current_thread_id,
                    )

                    # Check if thread has history
                    st.session_state.thread_has_history = check_thread_has_history(
                        st.session_state.current_user_id,
                        st.session_state.current_thread_id,
                    )

                    log(
                        f"Loaded thread history: {len(st.session_state.conversation_history)} entries"
                    )
                    log(f"Thread has history: {st.session_state.thread_has_history}")
        else:
            st.info("👆 Please select a user first")

    with col3:
        st.markdown("**Status**")
        if st.session_state.current_user_id and st.session_state.current_thread_id:
            st.success("✅ Ready")
            st.session_state.ids_configured = True
            st.markdown(f"**User:** `{st.session_state.current_user_id}`")
            st.markdown(f"**Thread:** `{st.session_state.current_thread_id}`")
            if st.session_state.thread_has_history:
                st.markdown(
                    f"**History:** {len(st.session_state.conversation_history)} entries"
                )
        else:
            st.warning("⚠️ Not Ready")
            st.session_state.ids_configured = False
            st.markdown("Select user & thread")

st.markdown("---")

# Developer mode checkbox
st.sidebar.checkbox("Developer Mode", value=False, key="developer_mode")

if st.session_state.get("developer_mode", False):
    st.sidebar.button("Clear Logs", on_click=lambda: st.session_state.logs.clear())
    with st.sidebar.expander("Logs", expanded=True):
        for message in st.session_state.logs[-10:]:  # Show last 10 logs
            st.sidebar.text(message)


def on_reset_click():
    st.session_state.user_query = ""
    st.session_state.response = ""
    st.session_state.submitted = False
    log("App state was reset (following user request).", print_to_sidebar=False)


def answer_query():
    if not st.session_state.ids_configured:
        st.error("❌ Please configure User ID and Thread ID first!")
        return

    if st.session_state.user_query.strip():
        log(f"User asked: {st.session_state.user_query}", print_to_sidebar=False)
        st.session_state.submitted = True
    else:
        log("User submitted an empty question.", print_to_sidebar=False)
        st.warning("Please enter a question before submitting.")


# Main interface - only enabled if IDs are configured
if st.session_state.ids_configured:

    # Display conversation history if it exists
    if st.session_state.conversation_history:
        display_conversation_history(st.session_state.conversation_history)
        st.markdown("### 💬 Continue Conversation")
    else:
        st.markdown("### 💬 Start New Conversation")

    with st.form("user_form", clear_on_submit=False, border=False):
        st.text_area(
            "Ask a question about the dataset:",
            key="user_query",
            height=100,
            value=st.session_state.get("user_query", ""),
            disabled=st.session_state.submitted,
        )
        st.form_submit_button(
            "Answer my question",
            disabled=st.session_state.submitted,
            on_click=answer_query,
        )

    if st.session_state.submitted:
        if not st.session_state.response:
            with st.spinner("Processing your question..."):
                try:
                    output = process_user_query(
                        st.session_state.user_query,
                        log,
                        user_id=st.session_state.current_user_id,
                        thread_id=st.session_state.current_thread_id,
                        has_history=st.session_state.thread_has_history,
                    )

                    st.session_state.response = output["response"]
                    log(f"Generated response: '{st.session_state.response}'")

                    # Save conversation to database
                    save_success = save_conversation_entry(
                        st.session_state.current_user_id,
                        st.session_state.current_thread_id,
                        st.session_state.user_query,
                        st.session_state.response,
                    )

                    if save_success:
                        # Reload conversation history to include new entry
                        st.session_state.conversation_history = load_thread_history(
                            st.session_state.current_user_id,
                            st.session_state.current_thread_id,
                        )
                        st.session_state.thread_has_history = True
                        log(
                            f"Query processed and saved successfully for user: {st.session_state.current_user_id}, thread: {st.session_state.current_thread_id}"
                        )

                except Exception as e:
                    st.error(f"❌ Error processing query: {str(e)}")
                    log(f"Error processing query: {str(e)}")
                    st.session_state.submitted = False

        if st.session_state.response:
            st.markdown("### 💬 Latest Response")
            st.write(f"""{st.session_state.response}""")
            st.button("Ask a new question", on_click=on_reset_click)

else:
    # Show message when IDs are not configured
    st.info(
        "🔧 **Please configure your User ID and Thread ID above to start using the Data Analyst Agent.**"
    )

    st.markdown(
        """
    ### Instructions:
    1. **Select or create a User ID** from the dropdown
    2. **Select or create a Thread ID** for your conversation
    3. Once both are configured, you can start asking questions about the dataset
    
    ### About User IDs and Thread IDs:
    - **User ID**: Identifies you as a user of the system
    - **Thread ID**: Represents a conversation thread/session
    - Each thread maintains its own conversation history
    - You can have multiple threads per user
    - Selecting an existing thread will load its conversation history
    """
    )

# Developer mode session state viewer
if st.session_state.get("developer_mode", False):
    with st.expander("Session State", expanded=False):
        display_state = dict(st.session_state)
        # Don't show large objects in session state viewer
        if "data" in display_state:
            display_state["data"] = (
                f"<Dataset object loaded: {type(display_state['data'])}>"
            )
        st.json(display_state)
