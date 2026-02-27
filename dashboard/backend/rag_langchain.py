from __future__ import annotations

from pathlib import Path
from statistics import mean
from typing import Any

LANGCHAIN_AVAILABLE = False

try:
    from langchain_community.retrievers import BM25Retriever
    from langchain_core.documents import Document

    LANGCHAIN_AVAILABLE = True
except Exception:
    BM25Retriever = None  # type: ignore[assignment]
    Document = None  # type: ignore[assignment]


class TaAIRAG:
    def __init__(self, base_dir: Path) -> None:
        self.base_dir = base_dir
        self.retriever: Any = None
        self.docs: list[Any] = []

    def _doc(self, text: str, source: str) -> Any:
        if LANGCHAIN_AVAILABLE and Document is not None:
            return Document(page_content=text, metadata={"source": source})
        return {"page_content": text, "metadata": {"source": source}}

    def _load_md_docs(self) -> list[Any]:
        out: list[Any] = []
        docs_dir = self.base_dir / "docs"
        if not docs_dir.exists():
            return out
        for p in sorted(docs_dir.glob("*.md")):
            try:
                content = p.read_text(encoding="utf-8")
            except Exception:
                continue
            if content.strip():
                out.append(self._doc(content, f"docs/{p.name}"))
        return out

    def _store_docs(self, rows: list[dict[str, Any]]) -> list[Any]:
        grouped: dict[int, list[dict[str, Any]]] = {}
        for r in rows:
            sid = int(r.get("Store", 0))
            if sid > 0:
                grouped.setdefault(sid, []).append(r)

        out: list[Any] = []
        for sid, vals in sorted(grouped.items()):
            vals = sorted(vals, key=lambda x: str(x.get("Date", "")))
            sales = [float(v.get("Weekly_Sales", 0.0)) for v in vals]
            preds = [float(v.get("Predicted_Sales", 0.0)) for v in vals]
            holidays = [float(v.get("Weekly_Sales", 0.0)) for v in vals if int(v.get("Holiday_Flag", 0)) == 1]
            non_holidays = [float(v.get("Weekly_Sales", 0.0)) for v in vals if int(v.get("Holiday_Flag", 0)) == 0]
            if not sales:
                continue
            avg_sales = mean(sales)
            avg_pred = mean(preds) if preds else 0.0
            peak = max(vals, key=lambda x: float(x.get("Weekly_Sales", 0.0)))
            low = min(vals, key=lambda x: float(x.get("Weekly_Sales", 0.0)))
            holiday_lift = (
                ((mean(holidays) - mean(non_holidays)) / max(mean(non_holidays), 1.0)) * 100.0
                if holidays and non_holidays
                else 0.0
            )
            txt = (
                f"Store {sid} summary.\n"
                f"Window: {vals[0].get('Date')} to {vals[-1].get('Date')} ({len(vals)} weeks).\n"
                f"Average weekly sales: {avg_sales:,.2f}. Average baseline prediction: {avg_pred:,.2f}.\n"
                f"Peak week: {peak.get('Date')} with sales {float(peak.get('Weekly_Sales', 0.0)):,.2f}.\n"
                f"Lowest week: {low.get('Date')} with sales {float(low.get('Weekly_Sales', 0.0)):,.2f}.\n"
                f"Holiday lift vs non-holiday: {holiday_lift:+.2f}%."
            )
            out.append(self._doc(txt, f"store/{sid}"))
        return out

    def _global_doc(self, rows: list[dict[str, Any]], coeffs: dict[str, float], model_info: dict[str, Any]) -> list[Any]:
        sales = [float(r.get("Weekly_Sales", 0.0)) for r in rows]
        preds = [float(r.get("Predicted_Sales", 0.0)) for r in rows]
        if sales:
            text = (
                "Global Walmart dashboard context.\n"
                f"Rows loaded: {len(rows)}. Average sales: {mean(sales):,.2f}. "
                f"Average baseline prediction: {mean(preds):,.2f}.\n"
                f"Model source: {model_info.get('model_source','unknown')}.\n"
                f"Coefficient target transform: {model_info.get('coef_target_transform','unknown')}.\n"
                f"Feature coefficients: "
                + ", ".join([f"{k}={float(v):+.6f}" for k, v in coeffs.items()])
            )
        else:
            text = "Global Walmart dashboard context is empty."
        return [self._doc(text, "global/context")]

    def refresh(self, rows: list[dict[str, Any]], coeffs: dict[str, float], model_info: dict[str, Any]) -> None:
        docs: list[Any] = []
        docs.extend(self._load_md_docs())
        docs.extend(self._global_doc(rows, coeffs, model_info))
        docs.extend(self._store_docs(rows))
        self.docs = docs
        if LANGCHAIN_AVAILABLE and BM25Retriever is not None and docs:
            try:
                self.retriever = BM25Retriever.from_documents(docs)
                self.retriever.k = 4
            except Exception:
                self.retriever = None
        else:
            self.retriever = None

    def _simple_retrieve(self, query: str, k: int = 4) -> list[str]:
        q = query.lower()
        scored: list[tuple[int, str]] = []
        for d in self.docs:
            text = d.page_content if hasattr(d, "page_content") else str(d.get("page_content", ""))
            t = text.lower()
            score = sum(1 for token in q.split() if token and token in t)
            if "store" in q and "store " in t:
                score += 2
            if score > 0:
                scored.append((score, text))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [t for _, t in scored[:k]]

    def retrieve(self, query: str, k: int = 4) -> list[str]:
        if self.retriever is not None:
            try:
                docs = self.retriever.invoke(query) or []
                out: list[str] = []
                for d in docs[:k]:
                    txt = d.page_content if hasattr(d, "page_content") else str(d)
                    if txt:
                        out.append(txt[:900])
                if out:
                    return out
            except Exception:
                pass
        return self._simple_retrieve(query, k=k)
