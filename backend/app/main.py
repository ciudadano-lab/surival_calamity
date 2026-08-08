import asyncio
import json
import re
from pathlib import Path

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

from game_data import VALID_PLAYERS, build_system_prompt
from rooms_store import init_db, create_room, get_room, save_room, room_exists, normalize_room_state
from llm_client import call_narrator
from ws_manager import manager

app = FastAPI(title="Zombie Sandbox Online")

frontend_dir = Path(__file__).resolve().parent.parent / "frontend"

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten before hosting this anywhere public
    allow_methods=["*"],
    allow_headers=["*"],
)

init_db()


@app.get("/", include_in_schema=False)
def serve_frontend():
    return FileResponse(frontend_dir / "index.html")


@app.get("/api/rooms/{code}", include_in_schema=False)
def api_get_room(code: str):
    state = get_room(code.upper())
    if state is None:
        raise HTTPException(404, "Room not found")
    return {"code": code.upper(), "state": state}


# One lock per room so concurrent rooms don't block each other, but two
# actions inside the SAME room (e.g. two people submitting at once) still
# resolve one at a time against consistent state.
_room_locks: dict[str, asyncio.Lock] = {}


def get_room_lock(room_code: str) -> asyncio.Lock:
    if room_code not in _room_locks:
        _room_locks[room_code] = asyncio.Lock()
    return _room_locks[room_code]


MENTION_RE = re.compile(r"player\s*#?\s*(\d+)", re.IGNORECASE)


def check_unknown_mention(text: str) -> str | None:
    """Non-blocking heads-up only — see game_data SHARED_RULES point 8.
    Mentioning an existing player never redirects the message; only a
    genuinely nonexistent number (4+) gets a note appended."""
    m = MENTION_RE.search(text)
    if not m:
        return None
    num = int(m.group(1))
    if num in VALID_PLAYERS:
        return None
    return f'There\'s no profile for "{m.group(0)}" in this world — only Player 1, 2, and 3 exist.'


def parse_reply(raw_text: str):
    match = re.search(r"```json\s*([\s\S]*?)```", raw_text)
    if not match:
        return raw_text.strip(), None
    narrative = raw_text[: match.start()].strip()
    try:
        payload = json.loads(match.group(1))
    except json.JSONDecodeError:
        payload = None
    return narrative, payload


def merge_state(room_state: dict, player_num: int, payload: dict | None):
    if not payload:
        return
    st = room_state["players"][str(player_num)]
    facts_delta = payload.get("facts_delta")
    if facts_delta:
        st["facts"].update(facts_delta)
    threads = payload.get("threads")
    if isinstance(threads, list):
        for update in threads:
            existing = next((t for t in st["threads"] if t["id"] == update.get("id")), None)
            if existing:
                existing.update(update)
            else:
                st["threads"].append(update)
    recap_line = payload.get("recap_line")
    if recap_line:
        st["recap"].append(recap_line)
        st["recap"] = st["recap"][-10:]


# ---------------------------------------------------------------------------
# REST: creating / checking rooms
# ---------------------------------------------------------------------------

@app.post("/api/rooms")
def api_create_room():
    code, state = create_room()
    return {"code": code, "state": state}


class SaveRoomRequest(BaseModel):
    state: dict
    saved_by_slot: int | None = None


@app.post("/api/rooms/{code}/save", include_in_schema=False)
def api_save_room(code: str, payload: SaveRoomRequest):
    room_code = code.upper()
    if not room_exists(room_code):
        raise HTTPException(404, "Room not found")

    state = normalize_room_state(payload.state)
    state.setdefault("meta", {})
    state["meta"].update({
        "saved": True,
        "save_count": int(state.get("meta", {}).get("save_count", 0)) + 1,
        "saved_by_slot": payload.saved_by_slot,
        "saved_at": state.get("meta", {}).get("saved_at") or "saved",
    })
    save_room(room_code, state)
    return {"code": room_code, "saved": True, "state": state}


@app.get("/api/rooms/{code}/load", include_in_schema=False)
def api_load_room(code: str):
    room_code = code.upper()
    state = get_room(room_code)
    if state is None:
        raise HTTPException(404, "Room not found")
    return {"code": room_code, "saved": bool(state.get("meta", {}).get("saved", False)), "state": state}


# ---------------------------------------------------------------------------
# WebSocket: live gameplay + chat
# ---------------------------------------------------------------------------

@app.websocket("/ws/{room_code}")
async def room_ws(websocket: WebSocket, room_code: str):
    room_code = room_code.upper()
    if not room_exists(room_code):
        await websocket.close(code=4404)
        return

    await manager.connect(room_code, websocket)
    try:
        state = get_room(room_code)
        await websocket.send_json({"type": "state", "state": state})

        while True:
            msg = await websocket.receive_json()
            mtype = msg.get("type")

            if mtype == "claim_slot":
                await handle_claim_slot(room_code, websocket, msg)
            elif mtype == "chat":
                await handle_chat(room_code, websocket, msg)
            elif mtype == "decision":
                await handle_decision(room_code, websocket, msg)
            else:
                await websocket.send_json({"type": "error", "message": f"Unknown message type: {mtype}"})

    except WebSocketDisconnect:
        pass
    finally:
        slot = manager.slot_for(room_code, websocket)
        manager.disconnect(room_code, websocket)
        if slot is not None:
            # Free the slot so a reconnect (or someone else) can take it.
            async with get_room_lock(room_code):
                state = get_room(room_code)
                if state:
                    state["occupied"][str(slot)] = False
                    save_room(room_code, state)
                    await manager.broadcast(room_code, {"type": "state", "state": state})


async def handle_claim_slot(room_code: str, websocket: WebSocket, msg: dict):
    slot = msg.get("slot")
    if slot not in VALID_PLAYERS:
        await websocket.send_json({"type": "error", "message": "Invalid player slot."})
        return

    async with get_room_lock(room_code):
        state = get_room(room_code)
        if state["occupied"].get(str(slot)):
            await websocket.send_json({"type": "error", "message": f"Player {slot} is already taken."})
            return
        state["occupied"][str(slot)] = True
        save_room(room_code, state)
        manager.claim_slot(room_code, websocket, slot)
        await websocket.send_json({"type": "slot_claimed", "slot": slot})
        await manager.broadcast(room_code, {"type": "state", "state": state})


async def handle_chat(room_code: str, websocket: WebSocket, msg: dict):
    slot = manager.slot_for(room_code, websocket)
    text = (msg.get("text") or "").strip()
    if slot is None:
        await websocket.send_json({"type": "error", "message": "Claim a player slot before chatting."})
        return
    if not text:
        return

    async with get_room_lock(room_code):
        state = get_room(room_code)
        entry = {"slot": slot, "text": text}
        state["chat"].append(entry)
        state["chat"] = state["chat"][-200:]  # cap history, this is a chat log not a database
        save_room(room_code, state)
        await manager.broadcast(room_code, {"type": "chat", "entry": entry})


async def handle_decision(room_code: str, websocket: WebSocket, msg: dict):
    slot = manager.slot_for(room_code, websocket)
    text = (msg.get("text") or "").strip()
    requested_slot = msg.get("slot")

    if slot is None:
        await websocket.send_json({"type": "error", "message": "Claim a player slot before playing."})
        return
    if requested_slot != slot:
        await websocket.send_json({"type": "error", "message": "You can only act as your own player."})
        return
    if not text:
        return

    target_slot = msg.get("target_slot")
    if target_slot is not None:
        if target_slot not in VALID_PLAYERS:
            await websocket.send_json({"type": "error", "message": "Invalid target player."})
            return
        if target_slot == slot:
            await websocket.send_json({"type": "error", "message": "You cannot interact with yourself."})
            return

    async with get_room_lock(room_code):
        state = get_room(room_code)
        player_state = state["players"][str(slot)]
        player_state["log"].append({"role": "player", "text": text})

        if target_slot is not None:
            state.setdefault("interactions", [])
            state["interactions"].append({
                "actor": slot,
                "target": target_slot,
                "text": text,
            })
            state["interactions"] = state["interactions"][-20:]
            player_state["log"].append({
                "role": "system",
                "text": f"You attempt to interact with Player {target_slot}.",
            })

        unknown_note = check_unknown_mention(text)
        if unknown_note:
            player_state["log"].append({"role": "system", "text": unknown_note})

        system_prompt = build_system_prompt(slot, state, interaction_target=target_slot)

        try:
            raw = await call_narrator(system_prompt, text)
        except RuntimeError as e:
            player_state["log"].append({"role": "system", "text": str(e)})
            save_room(room_code, state)
            await manager.broadcast(room_code, {"type": "state", "state": state})
            return

        narrative, payload = parse_reply(raw)
        merge_state(state, slot, payload)
        player_state["log"].append({"role": "narrator", "text": narrative or "(empty response)"})

        save_room(room_code, state)
        await manager.broadcast(room_code, {"type": "state", "state": state})
