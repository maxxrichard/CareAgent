"""Command-line interface: `careagent serve | chat | build-index | eval | seed`."""

from __future__ import annotations

import json

import typer
from rich.console import Console
from rich.panel import Panel

from careagent.config import get_settings

app = typer.Typer(help="CareAgent – agentic customer self-service assistant", no_args_is_help=True)
console = Console()


@app.command()
def serve(host: str = "0.0.0.0", port: int = 8000, reload: bool = False) -> None:
    """Run the REST API."""
    import uvicorn

    uvicorn.run("careagent.api.app:app", host=host, port=port, reload=reload)


@app.command()
def chat(
    message: list[str] = typer.Argument(None, help="Message to send; omit for interactive mode"),
    email: str | None = typer.Option(None, help="Authenticated customer email"),
    backend: str | None = typer.Option(None, help="anthropic | openai | ollama"),
    lightweight: bool = typer.Option(False, help="Use dependency-free embedder/reranker"),
) -> None:
    """Talk to the agent from the terminal."""
    from careagent.agent import CareAgentService

    svc = CareAgentService.create(get_settings(), backend=backend, lightweight_rag=lightweight)
    session = None

    def ask(text: str) -> None:
        nonlocal session
        resp = svc.answer_sync(text, session_id=session, customer_email=email)
        session = resp.session_id
        body = resp.answer
        if resp.citations:
            body += "\n\n[dim]Sources: " + ", ".join(c.source_id for c in resp.citations) + "[/dim]"
        if resp.escalated and resp.escalation:
            body += f"\n[yellow]→ handed off to a human ({resp.escalation.ticket_id})[/yellow]"
        console.print(
            Panel(
                body,
                title="CareAgent",
                subtitle=(
                    f"{resp.latency_ms:.0f} ms · €{resp.cost_eur:.4f} · grounding {resp.grounding.score:.2f}"
                ),
            )
        )

    if message:
        ask(" ".join(message))
        return
    console.print("[bold]CareAgent[/bold] interactive mode – Ctrl-C to exit")
    while True:
        try:
            ask(console.input("[cyan]you>[/cyan] "))
        except (KeyboardInterrupt, EOFError):
            break


@app.command("build-index")
def build_index(lightweight: bool = False) -> None:
    """Chunk and embed the policy documents and persist the vector index."""
    from careagent.rag import PolicyRetriever

    settings = get_settings()
    retriever = PolicyRetriever.build(settings, lightweight=lightweight)
    retriever.index.save(settings.index_dir)
    console.print(f"Indexed {len(retriever.index)} chunks → {settings.index_dir}")


@app.command()
def seed(n: int | None = None) -> None:
    """Create the synthetic orders database."""
    from careagent.data.orders import OrderStore

    store = OrderStore.seeded(n=n)
    console.print(f"{store.count()} orders in {store.db_path}")


@app.command("eval")
def evaluate(
    backends: str = typer.Option("anthropic", help="Comma-separated backends to compare"),
    limit: int | None = typer.Option(None, help="Only run the first N labelled queries"),
    judge: bool = typer.Option(True, help="Use the LLM judge for hallucination labels"),
    lightweight: bool = typer.Option(False, help="Dependency-free RAG (CI mode)"),
    output: str = typer.Option("data/eval/results", help="Directory for result files"),
) -> None:
    """Run the evaluation harness and log to MLflow."""
    from careagent.evaluation.runner import run_evaluation

    summary = run_evaluation(
        backends=[b.strip() for b in backends.split(",")],
        limit=limit,
        use_judge=judge,
        lightweight_rag=lightweight,
        output_dir=output,
    )
    console.print_json(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    app()
