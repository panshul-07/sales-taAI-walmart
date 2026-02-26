from sqlalchemy import Column, Integer, Float, String, Date, DateTime, Text, func, Index

from app.db.session import Base


class Sale(Base):
    __tablename__ = "sales"

    id = Column(Integer, primary_key=True, index=True)
    date = Column(Date, nullable=False, index=True)
    store_id = Column(Integer, nullable=False, index=True)
    product_id = Column(String(64), nullable=True)
    category = Column(String(128), nullable=True, index=True)
    quantity = Column(Float, nullable=False)
    revenue = Column(Float, nullable=False)
    cost = Column(Float, nullable=False)
    profit = Column(Float, nullable=False)


class Store(Base):
    __tablename__ = "stores"

    store_id = Column(Integer, primary_key=True)
    location = Column(String(128), nullable=True)
    size = Column(String(64), nullable=True)
    store_type = Column(String(64), nullable=True)
    region = Column(String(64), nullable=True, index=True)


class ChatMessage(Base):
    __tablename__ = "chat_messages"

    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(String(128), nullable=False, index=True)
    role = Column(String(16), nullable=False)
    content = Column(Text, nullable=False)
    meta_json = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)


Index("idx_sales_store_date", Sale.store_id, Sale.date)
