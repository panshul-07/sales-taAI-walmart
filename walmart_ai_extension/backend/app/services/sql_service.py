from sqlalchemy import text
from sqlalchemy.orm import Session

SAFE_SELECT_PREFIXES = ("select", "with")


class SQLService:
    def run_query(self, db: Session, sql: str, limit: int = 500) -> list[dict]:
        sql_clean = sql.strip().lower()
        if not sql_clean.startswith(SAFE_SELECT_PREFIXES):
            raise ValueError("Only SELECT queries are allowed")

        limited_sql = f"{sql.rstrip(';')} LIMIT {limit}"
        result = db.execute(text(limited_sql))
        cols = result.keys()
        return [dict(zip(cols, row)) for row in result.fetchall()]
