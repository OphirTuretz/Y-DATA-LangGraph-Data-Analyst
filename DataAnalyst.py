import streamlit as st
from data import Dataset
from engine import process_user_query
from datetime import datetime
from app.const import DATE_TIME_PATTERN

# Page config
st.set_page_config(page_title="Data Analyst Agent", layout="centered")
st.title("🤖 Data Analyst Agent")


# Cache dataset loading
@st.cache_data(show_spinner="Loading dataset, please wait...")
def load_bitext_dataset():
    ds = Dataset()
    return ds


if "response" not in st.session_state:
    st.session_state.response = ""

if "submitted" not in st.session_state:
    st.session_state.submitted = False

if "logs" not in st.session_state:
    st.session_state.logs = []

# Developer mode checkbox
st.sidebar.checkbox("Developer Mode", value=False, key="developer_mode")

if st.session_state.developer_mode:
    st.sidebar.button("Clear Logs", on_click=lambda: st.session_state.logs.clear())
    for message in st.session_state.logs:
        st.sidebar.text(message)


def log(message, print_to_sidebar=True):
    message = f"[{datetime.now().strftime(DATE_TIME_PATTERN)}] {message}"
    st.session_state.logs.append(message)
    print(message)

    if st.session_state.developer_mode and print_to_sidebar:
        st.sidebar.text(message)


# Initialize session state for data and messages
if "data" not in st.session_state:
    st.session_state.data = load_bitext_dataset()
    log("Dataset loaded and cached.")


def on_reset_click():
    st.session_state.data = load_bitext_dataset()
    st.session_state.user_query = ""
    st.session_state.response = ""
    st.session_state.submitted = False

    log("App state was reset (follwing user request).", print_to_sidebar=False)


def answer_query():
    if st.session_state.user_query.strip():
        log(f"User asked: {st.session_state.user_query}", print_to_sidebar=False)
        st.session_state.submitted = True
    else:
        log("User submitted an empty question.", print_to_sidebar=False)
        st.warning("Please enter a question before submitting.")


with st.form("user_form", clear_on_submit=False, border=False):
    st.text_area(
        "Ask a question about the dataset:",
        key="user_query",
        height=100,
        value=st.session_state.get("user_query", ""),
        disabled=st.session_state.submitted,
    )
    st.form_submit_button(
        "Answer my question", disabled=st.session_state.submitted, on_click=answer_query
    )


if st.session_state.submitted:
    if not st.session_state.response:
        with st.spinner("Processing your question..."):
            output = process_user_query(
                st.session_state.user_query,
                st.session_state.data,
                log,
            )

            st.session_state.data = output["dataset"]
            st.session_state.response = output["response"]
            log(f"Generated response: '{st.session_state.response}'")

    st.markdown("### 💬 Agent Response")
    st.write(f"""{st.session_state.response}""")

    st.button("Ask a new question", on_click=on_reset_click)


if st.session_state.developer_mode:
    with st.expander("Session State", expanded=False):
        st.json(dict(st.session_state))
