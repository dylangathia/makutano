"""Web server for the live console.

One shared Session runs in the background; every connected browser gets the
same picture over a WebSocket (about 20 frames a second, with the slower
panels once a second) and can send commands back. Static files in ./static
are the frontend.

    python -m makutano.live            # then open http://127.0.0.1:8000
"""
import asyncio
import json
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles

from .layout import describe
from .session import Session

STATIC = Path(__file__).parent / "static"
FRAME_S = 0.05       # 20 frames a second to the browser
FULL_EVERY = 20      # panels (forecast, KPIs, log) once a second


class Hub:
    def __init__(self, session):
        self.session = session
        self.clients = set()
        self.lock = asyncio.Lock()

    async def run(self):
        last, n = time.perf_counter(), 0
        while True:
            await asyncio.sleep(FRAME_S)
            now = time.perf_counter()
            async with self.lock:
                self.session.advance(now - last)
                frame = self.session.frame(full=n % FULL_EVERY == 0) if self.clients else None
            last = now
            n += 1
            if frame:
                await self.broadcast(json.dumps(frame, separators=(",", ":")))

    async def broadcast(self, text):
        for ws in list(self.clients):
            try:
                await ws.send_text(text)
            except Exception:
                self.clients.discard(ws)


def create_app(session=None):
    hub = Hub(session or Session())

    @asynccontextmanager
    async def lifespan(app):
        task = asyncio.create_task(hub.run())
        yield
        task.cancel()

    app = FastAPI(title="Makutano live", lifespan=lifespan)
    app.state.hub = hub

    @app.get("/api/layout")
    async def layout():
        return {**describe(), **hub.session.info()}

    @app.get("/api/state")
    async def state():
        async with hub.lock:
            return hub.session.frame(full=True)

    @app.post("/api/command")
    async def command(msg: dict):
        async with hub.lock:
            hub.session.command(msg)
            return hub.session.frame(full=True)

    @app.websocket("/ws")
    async def ws_endpoint(ws: WebSocket):
        await ws.accept()
        async with hub.lock:
            await ws.send_text(json.dumps(hub.session.frame(full=True)))
        hub.clients.add(ws)
        try:
            while True:
                msg = await ws.receive_json()
                async with hub.lock:
                    hub.session.command(msg)
                    full = hub.session.frame(full=True)
                await hub.broadcast(json.dumps(full, separators=(",", ":")))
        except (WebSocketDisconnect, RuntimeError, ValueError):
            pass
        finally:
            hub.clients.discard(ws)

    app.mount("/", StaticFiles(directory=STATIC, html=True), name="static")
    return app


app = create_app()
