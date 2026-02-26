from app.db.schemas import ChartSpec


class VisualizationService:
    def line_chart(self, rows: list[dict], x: str, y: str, title: str) -> ChartSpec:
        return ChartSpec(
            chart_type="line",
            title=title,
            data={
                "x": [r.get(x) for r in rows],
                "y": [r.get(y) for r in rows],
                "xField": x,
                "yField": y,
            },
        )

    def bar_chart(self, rows: list[dict], x: str, y: str, title: str) -> ChartSpec:
        return ChartSpec(
            chart_type="bar",
            title=title,
            data={
                "x": [r.get(x) for r in rows],
                "y": [r.get(y) for r in rows],
                "xField": x,
                "yField": y,
            },
        )

    def scatter_chart(self, rows: list[dict], x: str, y: str, title: str) -> ChartSpec:
        return ChartSpec(
            chart_type="scatter",
            title=title,
            data={
                "x": [r.get(x) for r in rows],
                "y": [r.get(y) for r in rows],
                "xField": x,
                "yField": y,
            },
        )
