from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session
from io import BytesIO
import pandas as pd
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

from app.db.session import get_db
from app.services.sql_service import SQLService

router = APIRouter()
sql_service = SQLService()


@router.get("/kpis")
def kpis(db: Session = Depends(get_db)) -> dict:
    rows = sql_service.run_query(
        db,
        "SELECT ROUND(SUM(revenue)::numeric,2) AS revenue, ROUND(SUM(profit)::numeric,2) AS profit, ROUND(AVG(revenue)::numeric,2) AS avg_revenue FROM sales",
        limit=1,
    )
    return rows[0] if rows else {"revenue": 0, "profit": 0, "avg_revenue": 0}


@router.get("/export/csv")
def export_csv(db: Session = Depends(get_db)) -> StreamingResponse:
    rows = sql_service.run_query(
        db,
        "SELECT date, store_id, product_id, category, quantity, revenue, cost, profit FROM sales ORDER BY date DESC",
        limit=5000,
    )
    df = pd.DataFrame(rows)
    buf = BytesIO()
    buf.write(df.to_csv(index=False).encode("utf-8"))
    buf.seek(0)
    return StreamingResponse(
        buf,
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=walmart_sales_export.csv"},
    )


@router.get("/export/excel")
def export_excel(db: Session = Depends(get_db)) -> StreamingResponse:
    rows = sql_service.run_query(
        db,
        "SELECT date, store_id, product_id, category, quantity, revenue, cost, profit FROM sales ORDER BY date DESC",
        limit=5000,
    )
    df = pd.DataFrame(rows)
    buf = BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="sales")
    buf.seek(0)
    return StreamingResponse(
        buf,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={"Content-Disposition": "attachment; filename=walmart_sales_export.xlsx"},
    )


@router.get("/export/pdf")
def export_pdf(db: Session = Depends(get_db)) -> StreamingResponse:
    row = sql_service.run_query(
        db,
        "SELECT ROUND(SUM(revenue)::numeric,2) AS total_revenue, ROUND(SUM(profit)::numeric,2) AS total_profit, COUNT(*) AS rows FROM sales",
        limit=1,
    )[0]
    buf = BytesIO()
    p = canvas.Canvas(buf, pagesize=letter)
    p.setFont("Helvetica-Bold", 14)
    p.drawString(50, 760, "Walmart Sales Summary Report")
    p.setFont("Helvetica", 11)
    p.drawString(50, 730, f"Total Revenue: {row['total_revenue']}")
    p.drawString(50, 712, f"Total Profit: {row['total_profit']}")
    p.drawString(50, 694, f"Rows Analyzed: {row['rows']}")
    p.save()
    buf.seek(0)
    return StreamingResponse(
        buf,
        media_type="application/pdf",
        headers={"Content-Disposition": "attachment; filename=walmart_sales_report.pdf"},
    )
