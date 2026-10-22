
import streamlit as st
from pipeline import run_research_pipeline

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
# CUSTOM CSS
# --------------------------------------------------

st.markdown("""
<style>
    .main-title {
        font-size: 2.5rem;
        font-weight: 800;
        margin-bottom: 0.2rem;
    }
    .subtitle {
        color: #9CA3AF;
        font-size: 1rem;
        margin-bottom: 1.5rem;
    }
    .stButton > button {
        width: 100%;
        border-radius: 10px;
        min-height: 3rem;
        font-weight: 600;
    }
    div[data-testid="stMetric"] {
        border: 1px solid rgba(128,128,128,0.25);
        padding: 1rem;
        border-radius: 12px;
    }
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# HEADER
# --------------------------------------------------

st.markdown(
    '<div class="main-title">🔬 Multi-Agent AI Researcher</div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="subtitle">'
    'Research smarter with AI-powered search, reading, writing, and evaluation.'
    '</div>',
    unsafe_allow_html=True,
)

# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

with st.sidebar:
    st.title("⚙️ Research Settings")
    st.write("Your research pipeline uses four AI stages.")

    st.markdown("---")
    st.markdown("### 🤖 Agents")
    st.write("🔎 Search Agent")
    st.write("📖 Reader Agent")
    st.write("✍️ Writer Chain")
    st.write("🧐 Critic Chain")

    st.markdown("---")
    st.markdown("### 💡 Suggested Topics")

    examples = [
        "Impact of war on stock market",
        "Artificial intelligence in education",
        "Inflation and the Indian economy",
        "Climate change and global markets",
        "Future of data analytics",
    ]

    selected_topic = st.selectbox(
        "Choose a sample topic",
        ["Select a topic..."] + examples,
    )

    if st.button("Use selected topic"):
        if selected_topic != "Select a topic...":
            st.session_state["research_topic"] = selected_topic

    st.markdown("---")
    st.caption("Powered by LangChain, Groq, Tavily, and Streamlit.")

# --------------------------------------------------
# TOPIC INPUT
# --------------------------------------------------

if "research_topic" not in st.session_state:
    st.session_state["research_topic"] = ""

topic = st.text_input(
    "📝 Enter your research topic",
    key="research_topic",
    placeholder="e.g. Impact of war on stock market",
)

run_button = st.button(
    "🚀 Start Research",
    type="primary",
    use_container_width=True,
)

# --------------------------------------------------
# RUN RESEARCH
# --------------------------------------------------

if run_button:
    if not topic.strip():
        st.warning("Please enter a research topic first.")
    else:
        try:
            with st.spinner(
                "AI agents are researching your topic. Please wait..."
            ):
                result = run_research_pipeline(topic.strip())

            st.session_state["research_result"] = result
            st.success("Research pipeline completed successfully!")

        except Exception as e:
            st.error("The research pipeline failed.")
            st.exception(e)

# --------------------------------------------------
# DISPLAY RESULTS
# --------------------------------------------------

if "research_result" in st.session_state:
    result = st.session_state["research_result"]

    st.markdown("---")
    st.header("📊 Research Results")

    report_tab, critique_tab, search_tab, reader_tab = st.tabs([
        "📄 Final Report",
        "🧐 Critic Evaluation",
        "🔎 Search Results",
        "📖 Reader Findings",
    ])

    with report_tab:
        st.subheader("Final Research Report")

        report = result.get("report", "No report generated.")

        st.markdown(report)

        st.download_button(
            label="⬇️ Download Research Report",
            data=report,
            file_name="research_report.md",
            mime="text/markdown",
            use_container_width=True,
        )

    with critique_tab:
        st.subheader("Critic Evaluation")

        critique = result.get(
            "critique",
            "No critic evaluation available.",
        )

        st.markdown(critique)

        st.download_button(
            label="⬇️ Download Evaluation",
            data=critique,
            file_name="research_evaluation.txt",
            mime="text/plain",
            use_container_width=True,
        )

    with search_tab:
        st.subheader("Web Search Findings")
        st.text(result.get("search_results", "No search results."))

    with reader_tab:
        st.subheader("Deep Reading Findings")
        st.text(result.get("reader_results", "No reader findings."))

else:
    st.markdown("---")

    st.subheader("✨ How it works")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown("### 🔎 Search")
        st.write("Finds relevant web information.")

    with col2:
        st.markdown("### 📖 Read")
        st.write("Scrapes a relevant page for deeper context.")

    with col3:
        st.markdown("### ✍️ Write")
        st.write("Creates a structured research report.")

    with col4:
        st.markdown("### 🧐 Critique")
        st.write("Evaluates the report and suggests improvements.")
