
from langchain.agents import create_agent
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
from tools import web_search, scrap_url
from dotenv import load_dotenv
from rich import print
import os

load_dotenv()

# Initialize the language model
llm = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0
)


# --------------------------------------------------
# AGENT BUILDERS
# --------------------------------------------------

def build_search_agent():
    return create_agent(
        model=llm,
        tools=[web_search],
    )


def build_reader_agent():
    return create_agent(
        model=llm,
        tools=[scrap_url],
    )


# --------------------------------------------------
# WRITER CHAIN
# --------------------------------------------------

writer_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are an expert research writer. "
        "Write clear, structured, insightful, and factual reports."
    ),
    (
        "human",
        """Write a detailed research report on the topic below.

Topic: {topic}

Research Gathered:
{research}

Structure the report as:
- Introduction
- Key Findings (minimum 3 well-explained points)
- Conclusion
- Sources (list all URLs found in the research)

Be detailed, factual, and professional.
Do not invent facts, statistics, or URLs."""
    ),
])

writer_chain = writer_prompt | llm | StrOutputParser()


# --------------------------------------------------
# CRITIC CHAIN
# --------------------------------------------------

critic_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a sharp and constructive research critic. "
        "Evaluate reports honestly and specifically."
    ),
    (
        "human",
        """Review the research report below and evaluate it strictly.

Report:
{report}

Respond in this exact format:

Score: X/10

Strengths:
- ...
- ...

Areas to Improve:
- ...
- ...

One line verdict:
..."""
    ),
])

critic_chain = critic_prompt | llm | StrOutputParser()


# --------------------------------------------------
# RESEARCH PIPELINE
# --------------------------------------------------

def run_research_pipeline(topic):
    state = {}

    # STEP 1: SEARCH AGENT
    print("\n[bold cyan]STEP 1: Search Agent[/bold cyan]")
    print("=" * 60)

    search_agent = build_search_agent()

    search_results = search_agent.invoke({
        "messages": [
            (
                "user",
                f"Search the web for recent, reliable information "
                f"about '{topic}'. Return relevant titles, URLs, "
                f"snippets, and useful findings."
            )
        ]
    })

    # Correctly access the final message
    state["search_results"] = (
        search_results["messages"][-1].content
    )

    print("\n[green]Search completed.[/green]")
    print(state["search_results"][:1500])


    # STEP 2: READER AGENT
    print("\n[bold cyan]STEP 2: Reader Agent[/bold cyan]")
    print("=" * 60)

    # Create the reader agent before using it
    reader_agent = build_reader_agent()

    reader_results = reader_agent.invoke({
        "messages": [
            (
                "user",
                f"""Research topic: {topic}

Here are the search results:
{state['search_results'][:8000]}

Choose the most relevant URL from these results.
Use the scrap_url tool to retrieve the page content.
Then summarize the useful information and include the
URL you actually scraped.

If scraping fails, explain the failure and use the
available search results without inventing page content."""
            )
        ]
    })

    state["reader_results"] = (
        reader_results["messages"][-1].content
    )

    print("\n[green]Reading completed.[/green]")
    print(state["reader_results"][:1500])


    # STEP 3: WRITER
    print("\n[bold cyan]STEP 3: Writer[/bold cyan]")
    print("=" * 60)

    research = (
        f"WEB SEARCH RESULTS:\n{state['search_results']}\n\n"
        f"DEEPER READING:\n{state['reader_results']}"
    )

    state["report"] = writer_chain.invoke({
        "topic": topic,
        "research": research,
    })

    print("\n[green]Research report generated.[/green]")
    print(state["report"])


    # STEP 4: CRITIC
    print("\n[bold cyan]STEP 4: Critic[/bold cyan]")
    print("=" * 60)

    state["critique"] = critic_chain.invoke({
        "report": state["report"],
    })

    print("\n[bold yellow]REPORT EVALUATION[/bold yellow]")
    print(state["critique"])


    # STEP 5: SAVE REPORT
    filename = "research_report.txt"

    with open(filename, "w", encoding="utf-8") as file:
        file.write(f"RESEARCH TOPIC: {topic}\n\n")
        file.write("=" * 60 + "\n")
        file.write("RESEARCH REPORT\n")
        file.write("=" * 60 + "\n\n")
        file.write(state["report"])
        file.write("\n\n")
        file.write("=" * 60 + "\n")
        file.write("CRITIC EVALUATION\n")
        file.write("=" * 60 + "\n\n")
        file.write(state["critique"])

    print(f"\n[bold green]Report saved to {filename}[/bold green]")

    return state


# --------------------------------------------------
# MAIN
# --------------------------------------------------

if __name__ == "__main__":
    topic = input("\nEnter a research topic: ").strip()

    if not topic:
        print("[red]Please enter a valid research topic.[/red]")
    else:
        run_research_pipeline(topic)
