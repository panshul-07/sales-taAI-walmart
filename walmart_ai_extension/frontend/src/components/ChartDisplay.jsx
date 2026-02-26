import Plot from 'react-plotly.js';

export default function ChartDisplay({ chart }) {
  if (!chart) return null;

  const x = chart.data?.x || [];
  const y = chart.data?.y || [];

  const typeMap = {
    line: 'scatter',
    bar: 'bar',
    scatter: 'scatter',
    heatmap: 'heatmap',
  };

  const mode = chart.chart_type === 'line' ? 'lines+markers' : 'markers';

  return (
    <div className="rounded-xl border border-slate-700 bg-slate-900 p-3">
      <h4 className="mb-2 text-sm font-semibold text-slate-200">{chart.title}</h4>
      <Plot
        data={[{ x, y, type: typeMap[chart.chart_type] || 'scatter', mode }]}
        layout={{
          paper_bgcolor: '#0f172a',
          plot_bgcolor: '#0f172a',
          font: { color: '#e2e8f0' },
          margin: { l: 40, r: 20, t: 20, b: 35 },
          autosize: true,
        }}
        useResizeHandler
        style={{ width: '100%', height: '320px' }}
        config={{ responsive: true }}
      />
    </div>
  );
}
