
from langchain.agents import create_agent
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
from tools import web_search, scrap_url
from dotenv import load_dotenv
from rich import print

load_dotenv()


# --------------------------------------------------
# LANGUAGE MODEL
# --------------------------------------------------

llm = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0,
    max_tokens=1200,
    max_retries=2,
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
        "Write clear, structured, factual, and insightful reports. "
        "Do not invent facts, statistics, or URLs."
    ),
    (
        "human",
        """Write a research report on the topic below.

Topic: {topic}

Research Gathered:
{research}

Use these sections:
1. Introduction
2. Key Findings (at least 3 explained points)
3. Conclusion
4. Sources (include URLs found in the research)

Be factual and professional. Keep the report concise but useful.
Distinguish verified facts from uncertain claims."""
    ),
])

writer_chain = writer_prompt | llm | StrOutputParser()


# --------------------------------------------------
# CRITIC CHAIN
# --------------------------------------------------

critic_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a constructive research critic. "
        "Evaluate the report honestly and specifically."
    ),
    (
        "human",
        """Review this research report:

{report}

Return the evaluation in this exact format:

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
# HELPER: EXTRACT FINAL MESSAGE
# --------------------------------------------------

def get_final_message(result):
    """Extract text from the final message returned by an agent."""
    messages = result.get("messages", [])

    for message in reversed(messages):
        if getattr(message, "type", "") == "ai":
            content = message.content

            if isinstance(content, str):
                return content

            if isinstance(content, list):
                return "\n".join(
                    str(item.get("text", ""))
                    for item in content
                    if isinstance(item, dict) and item.get("text")
                )

    return "No response was returned by the agent."


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
                f"Search for reliable information about '{topic}'. "
                "Use the web_search tool. Return a concise summary "
                "with relevant findings, titles, and source URLs. "
                "Avoid unnecessary detail."
            )
        ]
    })

    state["search_results"] = get_final_message(search_results)

    print("[green]Search completed.[/green]")
    print(state["search_results"][:1000])


    # STEP 2: READER AGENT
    print("\n[bold cyan]STEP 2: Reader Agent[/bold cyan]")
    print("=" * 60)

    reader_agent = build_reader_agent()

    reader_results = reader_agent.invoke({
        "messages": [
            (
                "user",
                f"""Research topic: {topic}

Search results:
{state['search_results'][:3000]}

Select the most relevant URL from the results.
Use the scrap_url tool to retrieve that page.
Summarize its most useful information in a concise format.
Include the URL actually scraped.

If scraping fails, explain the failure and do not invent
page content. Use only information available in the search results."""
            )
        ]
    })

    state["reader_results"] = get_final_message(reader_results)

    print("[green]Reading completed.[/green]")
    print(state["reader_results"][:1000])


    # STEP 3: WRITER
    print("\n[bold cyan]STEP 3: Writer[/bold cyan]")
    print("=" * 60)

    research = (
        "WEB SEARCH RESULTS:\n"
        f"{state['search_results'][:2500]}\n\n"
        "DEEPER READING:\n"
        f"{state['reader_results'][:2500]}"
    )

    state["report"] = writer_chain.invoke({
        "topic": topic,
        "research": research,
    })

    print("[green]Research report generated.[/green]")
    print(state["report"][:1500])


    # STEP 4: CRITIC
    print("\n[bold cyan]STEP 4: Critic[/bold cyan]")
    print("=" * 60)

    state["critique"] = critic_chain.invoke({
        "report": state["report"][:4000],
    })

    print("[bold yellow]REPORT EVALUATION[/bold yellow]")
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
