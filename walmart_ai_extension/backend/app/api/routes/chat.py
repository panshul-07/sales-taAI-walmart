from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from sqlalchemy import select, desc, func

from app.db.models import ChatMessage
from app.db.schemas import ChatRequest, ChatResponse
from app.db.session import get_db
from app.services.agent_service import AgentService

router = APIRouter()
agent = AgentService()


@router.post("/chat", response_model=ChatResponse)
def chat(payload: ChatRequest, db: Session = Depends(get_db)) -> ChatResponse:
    return agent.chat(db, payload.message, payload.session_id)


@router.get("/sessions")
def sessions(limit: int = 20, db: Session = Depends(get_db)) -> dict:
    sub = (
        select(ChatMessage.session_id, func.max(ChatMessage.created_at).label("last_at"))
        .group_by(ChatMessage.session_id)
        .subquery()
    )
    q = (
        select(ChatMessage.session_id, ChatMessage.content)
        .join(sub, (ChatMessage.session_id == sub.c.session_id))
        .where(ChatMessage.role == "user")
        .order_by(desc(sub.c.last_at))
        .limit(limit)
    )
    rows = db.execute(q).all()
    return {"sessions": [{"session_id": r.session_id, "preview": (r.content or "")[:80]} for r in rows]}


@router.get("/sessions/{session_id}")
def session_messages(session_id: str, db: Session = Depends(get_db)) -> dict:
    q = (
        select(ChatMessage.role, ChatMessage.content, ChatMessage.created_at)
        .where(ChatMessage.session_id == session_id)
        .order_by(ChatMessage.created_at.asc())
    )
    rows = db.execute(q).all()
    return {
        "session_id": session_id,
        "messages": [
            {"role": r.role, "content": r.content, "created_at": str(r.created_at)}
            for r in rows
        ],
    }
