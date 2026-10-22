
import os
import streamlit as st

# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Multi-Agent AI Researcher",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --------------------------------------------------
# LOAD API KEYS
# Supports local .env and Streamlit Cloud Secrets
# --------------------------------------------------

try:
    if "GROQ_API_KEY" in st.secrets:
        os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]

    if "TAVILY_API_KEY" in st.secrets:
        os.environ["TAVILY_API_KEY"] = st.secrets["TAVILY_API_KEY"]

except FileNotFoundError:
    # No Streamlit secrets file locally; .env is loaded
    # by pipeline.py and tools.py using python-dotenv.
    pass

# Import after loading deployment secrets
from pipeline import run_research_pipeline


# --------------------------------------------------
# CUSTOM CSS
# --------------------------------------------------

st.markdown(
    """
    <style>
        .main-title {
            font-size: 2.4rem;
            font-weight: 800;
            margin-bottom: 0.2rem;
        }

        .subtitle {
            color: #9CA3AF;
            font-size: 1rem;
            margin-bottom: 1.5rem;
        }

        div[data-testid="stMetric"] {
            border: 1px solid rgba(128, 128, 128, 0.25);
            padding: 1rem;
            border-radius: 12px;
        }

        .stButton > button {
            border-radius: 10px;
            font-weight: 600;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------
# SESSION STATE
# --------------------------------------------------

if "research_topic" not in st.session_state:
    st.session_state.research_topic = ""

if "research_result" not in st.session_state:
    st.session_state.research_result = None


# --------------------------------------------------
# HEADER
# --------------------------------------------------

st.markdown(
    '<div class="main-title">🔬 Multi-Agent AI Researcher</div>',
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="subtitle">
        Turn any research question into a structured report using
        AI-powered search, deep reading, writing, and critical evaluation.
    </div>
    """,
    unsafe_allow_html=True,
)

st.divider()


# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

with st.sidebar:
    st.title("⚙️ Research Studio")

    st.write("Your AI research workflow")

    st.markdown("---")

    st.markdown("### 🤖 Pipeline Stages")
    st.write("🔎 Search Agent — finds web information")
    st.write("📖 Reader Agent — scrapes a relevant page")
    st.write("✍️ Writer Chain — creates the report")
    st.write("🧐 Critic Chain — evaluates the report")

    st.markdown("---")

    st.markdown("### 💡 Suggested Topics")

    example_topics = [
        "Impact of war on stock market",
        "Inflation and the Indian economy",
        "Artificial intelligence in education",
        "Climate change and global markets",
        "Future of data analytics",
        "Effects of social media on students",
    ]

    selected_topic = st.selectbox(
        "Choose a sample topic",
        ["Choose a topic..."] + example_topics,
    )

    if st.button("Use this topic", use_container_width=True):
        if selected_topic != "Choose a topic...":
            st.session_state.research_topic = selected_topic
            st.rerun()

    st.markdown("---")

    st.caption(
        "Built with Streamlit, LangChain, Groq, and Tavily."
    )


# --------------------------------------------------
# TOPIC INPUT
# --------------------------------------------------

st.subheader("📝 What do you want to research?")

with st.form("research_form"):
    topic = st.text_input(
        "Research topic",
        key="research_topic",
        placeholder="Example: Impact of war on stock market",
        help="Enter a topic you want the AI agents to investigate.",
    )

    submitted = st.form_submit_button(
        "🚀 Start Research",
        type="primary",
        use_container_width=True,
    )


# --------------------------------------------------
# RUN PIPELINE
# --------------------------------------------------

if submitted:
    if not topic.strip():
        st.warning("Please enter a research topic first.")

    elif not os.getenv("GROQ_API_KEY") or not os.getenv("TAVILY_API_KEY"):
        st.error(
            "API keys are missing. Add GROQ_API_KEY and "
            "TAVILY_API_KEY to your local .env file or "
            "Streamlit Cloud Secrets."
        )

    else:
        st.session_state.research_result = None

        progress = st.progress(0, text="Preparing your research...")

        try:
            progress.progress(
                15,
                text="Running the search and reading agents...",
            )

            with st.spinner(
                "Researching your topic. This may take a few minutes..."
            ):
                result = run_research_pipeline(topic.strip())

            progress.progress(
                100,
                text="Research pipeline completed!",
            )

            st.session_state.research_result = result

            st.success("Your research report is ready!")

        except Exception as error:
            st.error("The research pipeline encountered an error.")
            st.code(str(error))

            with st.expander("Technical details"):
                st.exception(error)

        finally:
            progress.empty()


# --------------------------------------------------
# DISPLAY RESULTS
# --------------------------------------------------

result = st.session_state.research_result

if result:
    st.divider()
    st.header("📊 Research Workspace")

    st.write(
        "Topic: **"
        + str(topic if "topic" in locals() else st.session_state.research_topic)
        + "**"
    )

    report = result.get("report", "No report was generated.")
    critique = result.get("critique", "No critique is available.")
    search_results = result.get(
        "search_results",
        "No search results are available.",
    )
    reader_results = result.get(
        "reader_results",
        "No reader findings are available.",
    )

    report_tab, critique_tab, search_tab, reader_tab = st.tabs(
        [
            "📄 Final Report",
            "🧐 Critic Evaluation",
            "🔎 Search Results",
            "📖 Reader Findings",
        ]
    )

    # FINAL REPORT
    with report_tab:
        st.subheader("Research Report")

        st.markdown(report)

        st.download_button(
            label="⬇️ Download Report (.md)",
            data=report,
            file_name="research_report.md",
            mime="text/markdown",
            use_container_width=True,
            key="download_report",
        )

    # CRITIC EVALUATION
    with critique_tab:
        st.subheader("Critic Evaluation")

        st.markdown(critique)

        st.download_button(
            label="⬇️ Download Evaluation (.txt)",
            data=critique,
            file_name="research_evaluation.txt",
            mime="text/plain",
            use_container_width=True,
            key="download_critique",
        )

    # SEARCH RESULTS
    with search_tab:
        st.subheader("Web Search Findings")

        st.text(search_results)

        st.download_button(
            label="⬇️ Download Search Results",
            data=search_results,
            file_name="search_results.txt",
            mime="text/plain",
            key="download_search",
        )

    # READER FINDINGS
    with reader_tab:
        st.subheader("Deep Reading Findings")

        st.text(reader_results)

        st.download_button(
            label="⬇️ Download Reader Findings",
            data=reader_results,
            file_name="reader_findings.txt",
            mime="text/plain",
            key="download_reader",
        )

    st.divider()

    if st.button("🆕 Start a New Research Topic"):
        st.session_state.research_result = None
        st.session_state.research_topic = ""
        st.rerun()


# --------------------------------------------------
# EMPTY STATE
# --------------------------------------------------

else:
    st.divider()

    st.subheader("✨ How it works")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown("### 🔎 Search")
        st.write("Find relevant web sources.")

    with col2:
        st.markdown("### 📖 Read")
        st.write("Retrieve deeper page content.")

    with col3:
        st.markdown("### ✍️ Write")
        st.write("Generate a structured research report.")

    with col4:
        st.markdown("### 🧐 Critique")
        st.write("Score the report and identify improvements.")

    st.info(
        "Enter a research topic above to begin. "
        "Your report and evaluation will appear here."
    )
