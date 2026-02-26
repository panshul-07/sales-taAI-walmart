import csv
from datetime import datetime
from pathlib import Path

from sqlalchemy.orm import Session

from app.db.models import Sale
from app.db.session import SessionLocal


def parse_date(val: str):
    for fmt in ("%Y-%m-%d", "%d-%m-%Y", "%m-%d-%Y", "%d/%m/%Y"):
        try:
            return datetime.strptime(val, fmt).date()
        except ValueError:
            pass
    return None


def load(csv_path: Path) -> None:
    db: Session = SessionLocal()
    try:
        with csv_path.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for r in reader:
                dt = parse_date(str(r.get("date") or r.get("Date") or ""))
                if dt is None:
                    continue
                revenue = float(r.get("revenue") or r.get("Weekly_Sales") or 0)
                cost = float(r.get("cost") or 0)
                qty = float(r.get("quantity") or 1)
                profit = float(r.get("profit") or (revenue - cost))
                sale = Sale(
                    date=dt,
                    store_id=int(float(r.get("store_id") or r.get("Store") or 0)),
                    product_id=str(r.get("product_id") or "na"),
                    category=str(r.get("category") or "general"),
                    quantity=qty,
                    revenue=revenue,
                    cost=cost,
                    profit=profit,
                )
                db.add(sale)
        db.commit()
        print("Loaded CSV:", csv_path)
    finally:
        db.close()


if __name__ == "__main__":
    import sys

    path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/data/sales.csv")
    load(path)
