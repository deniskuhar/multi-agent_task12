from __future__ import annotations

import getpass
from typing import Any

from config import APP_TITLE, SEPARATOR, get_settings
from langfuse_utils import auth_check, flush_langfuse, get_langfuse_client, trace_attributes
from supervisor import (
    new_thread_id,
    run_supervisor,
    revise_report_with_feedback,
)
from tools import save_report

settings = get_settings()


def _handle_save_flow(report: dict[str, Any], thread_id: str) -> str:
    while True:
        print("\n" + SEPARATOR)
        print("  ACTION REQUIRES APPROVAL")
        print(SEPARATOR)
        print(f"  Tool:  save_report")
        print(f"  Filename:  {report['filename']}")
        preview = report["content"][:800]
        print(f"  Preview:\n{preview}")
        if len(report["content"]) > 800:
            print("\n  ...")

        decision = input("\n approve / edit / reject: ").strip().lower()

        if decision not in {"approve", "edit", "reject"}:
            print("Please enter approve, edit, or reject.")
            continue

        if decision == "approve":
            return save_report(filename=report["filename"], content=report["content"])

        if decision == "edit":
            feedback = input("✏️  Your feedback: ").strip()
            report = revise_report_with_feedback(
                report,
                feedback,
                session_id=thread_id,
                user_id=report.get("user_id"),
                tags=[*settings.langfuse_tags, "report-revision"],
            )
            print("\n[Supervisor] Report revised based on your feedback.")
            continue

        return "Report saving was cancelled."


def main() -> None:
    thread_id = new_thread_id()
    user_id = settings.langfuse_default_user_id or getpass.getuser() or "local-user"

    print(SEPARATOR)
    print(APP_TITLE)
    print("Type 'exit' or 'quit' to leave. Type 'new' to reset the session.")
    print(SEPARATOR)

    if auth_check():
        print(f"Langfuse tracing enabled for user_id={user_id!r}, session_id={thread_id!r}")
    else:
        print("Langfuse auth check failed. Verify LANGFUSE_* keys before running homework-12 screenshots.")

    while True:
        try:
            user_input = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not user_input:
            continue

        if user_input.lower() in {"exit", "quit"}:
            print("Goodbye!")
            break

        if user_input.lower() == "new":
            thread_id = new_thread_id()
            print(f"Started a new session: {thread_id}")
            continue

        try:
            langfuse = get_langfuse_client()
            with trace_attributes(
                session_id=thread_id,
                user_id=user_id,
                tags=[*settings.langfuse_tags, "interactive-run"],
                metadata={"environment": settings.langfuse_environment},
            ):
                with langfuse.start_as_current_observation(as_type="agent", name="interactive-turn") as turn_obs:
                    turn_obs.update(input=user_input)
                    report = run_supervisor(
                        user_input,
                        session_id=thread_id,
                        user_id=user_id,
                        tags=[*settings.langfuse_tags, "interactive-run"],
                    )

                    print("\n" + SEPARATOR)
                    print("Draft report prepared")
                    print(SEPARATOR)
                    print(f"Filename: {report['filename']}")
                    preview = report["content"][:1500]
                    print(preview)
                    if len(report["content"]) > 1500:
                        print("\n...")

                    final_message = _handle_save_flow(report, thread_id)
                    turn_obs.update(output=final_message)
                    print(f"\nAgent: {final_message}")
        except Exception as exc:
            print(f"\nAgent error: {exc}")
        finally:
            flush_langfuse()


if __name__ == "__main__":
    main()
