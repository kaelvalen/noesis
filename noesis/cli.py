"""Command-line interface for NOESIS."""

import argparse
import sys

from noesis.core.engine import NoesisEngine


def cmd_init(args):
    engine = NoesisEngine(model_path=args.model, use_mock=args.mock)
    engine.save()
    print("NOESIS initialized.")
    print(engine.hw.summary())


def cmd_chat(args):
    engine = NoesisEngine(model_path=args.model, use_mock=args.mock)
    print("NOESIS chat mode. Type 'exit' to quit, 'save' to checkpoint.")
    while True:
        try:
            user_input = input("> ")
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            break
        if user_input.strip().lower() in ("exit", "quit"):
            engine.save()
            print("Session saved.")
            break
        if user_input.strip().lower() == "save":
            engine.save()
            print("Saved.")
            continue
        if not user_input.strip():
            continue
        response = engine.interact(user_input, max_new_tokens=args.max_tokens)
        print(response)


def cmd_learn_web(args):
    engine = NoesisEngine(model_path=args.model, use_mock=args.mock)
    engine.ingest_web(args.urls)
    engine.save()
    print(f"Ingested {len(args.urls)} URL(s).")


def cmd_learn_files(args):
    engine = NoesisEngine(model_path=args.model, use_mock=args.mock)
    engine.ingest_files(args.paths)
    engine.save()
    print(f"Ingested file patterns: {args.paths}")


def cmd_status(args):
    engine = NoesisEngine(model_path=args.model, use_mock=args.mock)
    for key, value in engine.status().items():
        print(f"{key}: {value}")


def cmd_save(args):
    engine = NoesisEngine(model_path=args.model, use_mock=args.mock)
    engine.save()
    print("Session saved.")


def cmd_consolidate(args):
    engine = NoesisEngine(model_path=args.model, use_mock=args.mock)
    path = engine.consolidate()
    if path:
        print(f"Consolidated new expert saved to {path}")
    else:
        print("Consolidation skipped (not enough traces).")


def build_parser():
    parser = argparse.ArgumentParser(
        prog="noesis",
        description="NOESIS — memory-centric, self-improving inference system",
    )
    parser.add_argument(
        "--model",
        default="BlinkDL/rwkv-5-world-1b5",
        help="Backbone model path or HF id",
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Use a tiny mock backbone for testing (no large model download)",
    )

    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("init", help="Initialize NOESIS state directories")

    chat = sub.add_parser("chat", help="Stateful chat loop")
    chat.add_argument(
        "--max-tokens", type=int, default=128, help="Max new tokens per turn"
    )

    learn_web = sub.add_parser("learn-web", help="Ingest web pages")
    learn_web.add_argument("urls", nargs="+", help="URLs to ingest")

    learn_files = sub.add_parser("learn-files", help="Ingest local files")
    learn_files.add_argument("paths", nargs="+", help="File glob patterns")

    sub.add_parser("status", help="Show system status")
    sub.add_parser("save", help="Manually save session state")
    sub.add_parser("consolidate", help="Trigger adapter training/consolidation")

    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)

    commands = {
        "init": cmd_init,
        "chat": cmd_chat,
        "learn-web": cmd_learn_web,
        "learn-files": cmd_learn_files,
        "status": cmd_status,
        "save": cmd_save,
        "consolidate": cmd_consolidate,
    }

    cmd = commands.get(args.command)
    if cmd is None:
        parser.print_help()
        sys.exit(1)
    cmd(args)


if __name__ == "__main__":
    main()
