import argparse
import uvicorn


def main():
    parser = argparse.ArgumentParser(description="WhatsApp Style AI Writing Assistant")
    parser.add_argument("--host", default="127.0.0.1", help="Host interface (default: 127.0.0.1 - strictly local)")
    parser.add_argument("--port", type=int, default=8000, help="Port to listen on (default: 8000)")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload for development")
    args = parser.parse_args()

    print("\n" + "=" * 65)
    print("  🚀 WhatsApp Style AI Assistant (Local-First)")
    print("=" * 65)
    print(f"  • Interface:       http://{args.host}:{args.port}")
    print("  • Privacy:         100% Local (no cloud transmission)")
    print("  • Human Review:    Mandatory for all generated drafts")
    print("=" * 65 + "\n")

    uvicorn.run("app.web.main:app", host=args.host, port=args.port, reload=args.reload)


if __name__ == "__main__":
    main()
