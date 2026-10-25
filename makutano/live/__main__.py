"""Run the live console: python -m makutano.live [--host 127.0.0.1] [--port 8000]"""
import argparse

import uvicorn


def main():
    ap = argparse.ArgumentParser(description="Makutano live console")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    a = ap.parse_args()
    print(f"Makutano live console: http://{a.host}:{a.port}")
    uvicorn.run("makutano.live.server:app", host=a.host, port=a.port, log_level="warning")


if __name__ == "__main__":
    main()
