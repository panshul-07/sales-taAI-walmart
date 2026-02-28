import { useEffect, useMemo, useState } from 'react';
import {
  ResponsiveContainer,
  LineChart,
  Line,
  BarChart,
  Bar,
  CartesianGrid,
  XAxis,
  YAxis,
  Tooltip,
  ScatterChart,
  Scatter,
} from 'recharts';
import { Button } from './components/ui/button';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from './components/ui/card';
import { Input } from './components/ui/input';
import { Select } from './components/ui/select';
import { Badge } from './components/ui/badge';
import { Separator } from './components/ui/separator';
import './styles.css';

const FEATURES = ['CPI', 'Unemployment', 'Fuel_Price', 'Temperature'];
const money = (n) => new Intl.NumberFormat('en-US', { style: 'currency', currency: 'USD', maximumFractionDigits: 0 }).format(Number(n || 0));
const moneyCompact = (n) => new Intl.NumberFormat('en-US', {
  style: 'currency',
  currency: 'USD',
  notation: 'compact',
  maximumFractionDigits: 1,
}).format(Number(n || 0));
const compact = (n) => {
  const v = Number(n || 0);
  if (Math.abs(v) >= 1_000_000) return `${(v / 1_000_000).toFixed(1)}M`;
  if (Math.abs(v) >= 1_000) return `${(v / 1_000).toFixed(1)}K`;
  return v.toFixed(0);
};
const api = (path, options) => fetch(path, options).then((r) => {
  if (!r.ok) throw new Error(`${path} failed`);
  return r.json();
});
const numericDomain = (values, minPad = 1, padRatio = 0.1) => {
  const xs = values.map((v) => Number(v)).filter((v) => Number.isFinite(v));
  if (!xs.length) return [0, 1];
  let min = Math.min(...xs);
  let max = Math.max(...xs);
  if (min === max) {
    min -= minPad;
    max += minPad;
  } else {
    const pad = Math.max(minPad, (max - min) * padRatio);
    min -= pad;
    max += pad;
  }
  return [min, max];
};

const fitLine = (points) => {
  const xs = points.map((p) => Number(p.x)).filter((v) => Number.isFinite(v));
  const ys = points.map((p) => Number(p.y)).filter((v) => Number.isFinite(v));
  const n = Math.min(xs.length, ys.length);
  if (n < 2) return { slope: 0, intercept: 0, corr: 0, r2: 0 };
  const mx = xs.reduce((a, b) => a + b, 0) / n;
  const my = ys.reduce((a, b) => a + b, 0) / n;
  let cov = 0;
  let vx = 0;
  let vy = 0;
  for (let i = 0; i < n; i += 1) {
    const dx = xs[i] - mx;
    const dy = ys[i] - my;
    cov += dx * dy;
    vx += dx * dx;
    vy += dy * dy;
  }
  const slope = vx ? cov / vx : 0;
  const intercept = my - slope * mx;
  const corr = (vx && vy) ? cov / Math.sqrt(vx * vy) : 0;
  return { slope, intercept, corr, r2: corr * corr };
};

const toGoogleAllDayRange = (dateStr) => {
  const d = new Date(`${dateStr}T00:00:00`);
  if (Number.isNaN(d.getTime())) return null;
  const next = new Date(d);
  next.setDate(next.getDate() + 1);
  const fmt = (x) => {
    const y = x.getFullYear();
    const m = String(x.getMonth() + 1).padStart(2, '0');
    const day = String(x.getDate()).padStart(2, '0');
    return `${y}${m}${day}`;
  };
  return `${fmt(d)}/${fmt(next)}`;
};

const isDateLike = (v) => /^\d{4}-\d{2}-\d{2}$/.test(String(v || ''));

function PointDot({ cx, cy, stroke = '#7fb6ff', payload, onPointClick }) {
  if (!Number.isFinite(cx) || !Number.isFinite(cy)) return null;
  return (
    <circle
      cx={cx}
      cy={cy}
      r={3.5}
      fill={stroke}
      stroke="rgba(2,10,16,.65)"
      strokeWidth={1}
      className="clickable-dot"
      onClick={(e) => {
        e.stopPropagation();
        onPointClick?.(payload);
      }}
    />
  );
}

export default function App() {
  const [theme, setTheme] = useState(localStorage.getItem('theme') || 'dark');
  const [store, setStore] = useState('all');
  const [weeks, setWeeks] = useState(160);
  const [factor, setFactor] = useState('CPI');
  const [stores, setStores] = useState([]);
  const [rows, setRows] = useState([]);
  const [corr, setCorr] = useState([]);
  const [coef, setCoef] = useState([]);
  const [status, setStatus] = useState('Loading...');
  const [overview, setOverview] = useState(null);
  const [adjustments, setAdjustments] = useState({ CPI: 0, Unemployment: 0, Fuel_Price: 0, Temperature: 0 });

  const [chatSessionId, setChatSessionId] = useState(localStorage.getItem('taai_session_id') || '');
  const [chatText, setChatText] = useState('');
  const [chatBusy, setChatBusy] = useState(false);
  const [chatMessages, setChatMessages] = useState([{ role: 'assistant', content: 'taAI ready. Ask economic questions about the data.' }]);
  const [chatSessions, setChatSessions] = useState([]);
  const [insights, setInsights] = useState(null);
  const [distStats, setDistStats] = useState(null);

  useEffect(() => {
    document.documentElement.setAttribute('data-theme', theme);
    localStorage.setItem('theme', theme);
  }, [theme]);

  const adjustedRows = useMemo(() => {
    const beta = Object.fromEntries(coef.map((c) => [c.feature, Number(c.beta_per_unit || 0)]));
    return rows.map((r) => {
      let delta = 0;
      const xAdj = {};
      for (const f of FEATURES) {
        const base = Number(r[f]);
        const pct = Number(adjustments[f] || 0) / 100;
        const newVal = base * (1 + pct);
        xAdj[f] = newVal;
        delta += (newVal - base) * (beta[f] || 0);
      }
      return {
        ...r,
        Adjusted_Factor: xAdj[factor],
        Simulated_Sales: Math.max(1000, Number(r.Predicted_Sales) + delta),
      };
    });
  }, [rows, coef, adjustments, factor]);

  const trendData = useMemo(() => rows.map((r, i) => ({
    Date: r.Date,
    Actual: Number(r.Weekly_Sales),
    Baseline: Number(r.Predicted_Sales),
    Simulated: Number(adjustedRows[i]?.Simulated_Sales || r.Predicted_Sales),
  })), [rows, adjustedRows]);

  const scatterData = useMemo(() => adjustedRows.map((r) => ({
    x: Number(r.Adjusted_Factor),
    y: Number(r.Simulated_Sales),
    baseline: Number(r.Predicted_Sales),
    actual: Number(r.Weekly_Sales),
    date: r.Date,
    holiday: Number(r.Holiday_Flag) === 1,
  })), [adjustedRows]);

  const scatterFit = useMemo(() => fitLine(scatterData), [scatterData]);
  const [scatterXMin, scatterXMax] = useMemo(() => numericDomain(scatterData.map((d) => d.x), 0.5, 0.14), [scatterData]);
  const [scatterYMin, scatterYMax] = useMemo(() => numericDomain(scatterData.map((d) => d.y), 1000, 0.12), [scatterData]);
  const scatterTrend = useMemo(() => (
    [
      { x: scatterXMin, y: (scatterFit.slope * scatterXMin) + scatterFit.intercept },
      { x: scatterXMax, y: (scatterFit.slope * scatterXMax) + scatterFit.intercept },
    ]
  ), [scatterXMin, scatterXMax, scatterFit]);

  const simulatedAvg = useMemo(() => adjustedRows.length ? adjustedRows.reduce((a, b) => a + Number(b.Simulated_Sales), 0) / adjustedRows.length : 0, [adjustedRows]);

  const openCalendarForPoint = (dateStr, title, details) => {
    const range = toGoogleAllDayRange(String(dateStr || ''));
    if (!range) return;
    const params = new URLSearchParams({
      action: 'TEMPLATE',
      text: title || 'taAI Sales Review',
      dates: range,
      details: details || 'Generated from taAI dashboard',
    });
    const calendarUrl = `https://calendar.google.com/calendar/render?${params.toString()}`;
    const popup = window.open(calendarUrl, '_blank', 'noopener,noreferrer');
    if (!popup) window.location.assign(calendarUrl);
  };

  const openCalendarForTrendPoint = (point) => {
    if (!isDateLike(point?.Date)) return;
    openCalendarForPoint(
      point.Date,
      `taAI Sales Review (${store === 'all' ? 'All Stores' : `Store ${store}`})`,
      `Actual=${money(point.Actual)}, Baseline=${money(point.Baseline)}, Simulated=${money(point.Simulated)}`,
    );
  };

  async function loadData() {
    setStatus('Loading...');
    const q = `store=${encodeURIComponent(store)}&weeks=${weeks}`;
    const [st, ov, sr, cr, cf] = await Promise.all([
      api('/api/stores'),
      api(`/api/overview?${q}`),
      api(`/api/store-data?${q}`),
      api(`/api/correlations?${q}`),
      api(`/api/coefficients?${q}`),
    ]);
    setStores(st);
    setOverview(ov);
    setRows(sr);
    setCorr(cr);
    setCoef(cf.rows || []);
    const label = cf.model_source === 'extra_trees_notebook_pickle' ? 'Notebook ExtraTrees predictions + notebook coefficients' : 'Model loaded';
    setStatus(`Showing ${ov.records} rows (${ov.date_min} to ${ov.date_max}) | ${label}`);
    try {
      const ins = await api(`/api/taai/insights?${q}`);
      setInsights(ins);
    } catch {
      setInsights(null);
    }
    try {
      const ds = await api(`/api/stats/distribution?${q}`);
      setDistStats(ds);
    } catch {
      setDistStats(null);
    }
  }

  async function loadChatSessions() {
    try {
      const res = await api('/api/taai/sessions?limit=10');
      setChatSessions(res.sessions || []);
    } catch {
      setChatSessions([]);
    }
  }

  async function loadChatSession(id) {
    if (!id) return;
    const res = await api(`/api/taai/sessions/${encodeURIComponent(id)}`);
    setChatSessionId(id);
    localStorage.setItem('taai_session_id', id);
    setChatMessages((res.messages || []).map((m) => ({ role: m.role, content: m.content })));
  }

  async function sendChat(message) {
    const msg = String(message || '').trim();
    if (!msg || chatBusy) return;
    setChatBusy(true);
    setChatMessages((m) => [...m, { role: 'user', content: msg }, { role: 'assistant', content: 'Analyzing…' }]);
    try {
      const res = await api('/api/taai/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message: msg, session_id: chatSessionId || null }),
      });
      const sid = res.session_id || chatSessionId;
      if (sid) {
        setChatSessionId(sid);
        localStorage.setItem('taai_session_id', sid);
      }
      setChatMessages((m) => {
        const next = [...m];
        if (next.length && next[next.length - 1].content === 'Analyzing…') next.pop();
        next.push({
          role: 'assistant',
          content: res.answer || 'No response',
          intent: res.intent || null,
          confidence: Number(res.confidence || 0),
          charts: Array.isArray(res.charts) ? res.charts : [],
        });
        return next;
      });
      loadChatSessions();
    } catch (e) {
      setChatMessages((m) => {
        const next = [...m];
        if (next.length && next[next.length - 1].content === 'Analyzing…') next.pop();
        next.push({ role: 'assistant', content: `Error: ${String(e)}` });
        return next;
      });
    } finally {
      setChatBusy(false);
      setChatText('');
    }
  }

  useEffect(() => {
    loadData().catch((e) => setStatus(`Failed: ${e.message}`));
  }, [store, weeks]);

  useEffect(() => {
    loadChatSessions();
    if (chatSessionId) loadChatSession(chatSessionId).catch(() => {});
  }, []);

  return (
    <div className="app-shell">
      <div className="bg-layer" />
      <div className="dashboard-wrap">
        <header className="hero">
          <div>
            <h1>Walmart Forecast Dashboard</h1>
            <p className="hero-sub">shadcn-style UX, notebook-linked forecasting, MCP-ready diagnostics, and calendar-aware chart interactions.</p>
          </div>
          <Card className="theme-card">
            <CardContent className="theme-row">
              <Badge variant="subtle">Theme</Badge>
              <Button variant={theme === 'dark' ? 'default' : 'outline'} onClick={() => setTheme('dark')}>Dark</Button>
              <Button variant={theme === 'light' ? 'default' : 'outline'} onClick={() => setTheme('light')}>Light</Button>
            </CardContent>
          </Card>
        </header>

        <section className="top-grid">
          <Card className="top-panel left">
            <CardContent className="field-grid">
              <div>
                <label className="field-label">Store</label>
                <Select value={store} onChange={(e) => setStore(e.target.value)}>
                  <option value="all">All Stores</option>
                  {stores.map((s) => <option key={s.Store} value={String(s.Store)}>Store {s.Store}</option>)}
                </Select>
              </div>
              <div>
                <label className="field-label">Weeks: {weeks}</label>
                <input type="range" min="52" max="260" value={weeks} onChange={(e) => setWeeks(Number(e.target.value))} />
              </div>
            </CardContent>
          </Card>

          <Card className="top-panel right">
            <CardContent>
              <label className="field-label">Scatter Factor</label>
              <Select value={factor} onChange={(e) => setFactor(e.target.value)}>
                {FEATURES.map((f) => <option key={f} value={f}>{f}</option>)}
              </Select>
              <p className="status-text">{status}</p>
            </CardContent>
          </Card>
        </section>

        <section className="metric-grid">
          <Card className="metric-card"><CardContent><p className="metric-key">Average Weekly Sales</p><p className="metric-val" title={money(overview?.avg_weekly_sales || 0)}>{moneyCompact(overview?.avg_weekly_sales || 0)}</p></CardContent></Card>
          <Card className="metric-card"><CardContent><p className="metric-key">Peak Sales</p><p className="metric-val" title={money(overview?.peak_sales || 0)}>{moneyCompact(overview?.peak_sales || 0)}</p></CardContent></Card>
          <Card className="metric-card"><CardContent><p className="metric-key">Holiday Avg</p><p className="metric-val" title={money(overview?.holiday_avg || 0)}>{moneyCompact(overview?.holiday_avg || 0)}</p></CardContent></Card>
          <Card className="metric-card"><CardContent><p className="metric-key">Holiday Count</p><p className="metric-val">{overview?.holiday_count ?? 0}</p></CardContent></Card>
          <Card className="metric-card"><CardContent><p className="metric-key">Simulated Avg Sales</p><p className="metric-val" title={money(simulatedAvg)}>{moneyCompact(simulatedAvg)}</p></CardContent></Card>
        </section>

        <Card className="chat-panel">
          <CardHeader className="chat-header">
            <div>
              <CardTitle>taAI</CardTitle>
              <CardDescription>Ask anything about this project and request charts. Click chart points to create calendar tasks.</CardDescription>
            </div>
            <Button variant="outline" onClick={() => { setChatSessionId(''); localStorage.removeItem('taai_session_id'); setChatMessages([{ role: 'assistant', content: 'New chat started.' }]); }}>New</Button>
          </CardHeader>
          <Separator />
          <CardContent className="chat-body">
            {chatMessages.map((m, i) => (
              <div key={i} className={`msg ${m.role === 'user' ? 'user' : 'bot'}`}>
                {m.content}
                {m.role !== 'user' && m.intent && (
                  <div className="msg-meta">
                    <Badge variant="subtle">{String(m.intent).replaceAll('_', ' ')}</Badge>
                    <span>{Math.round(Number(m.confidence || 0) * 100)}% confidence</span>
                  </div>
                )}
                {Array.isArray(m.charts) && m.charts.length > 0 && (
                  <div className="chat-charts">
                    {m.charts.map((c, ci) => (
                      <Card key={ci} className="chat-chart-card">
                        <CardContent>
                          <p className="field-label">{c.title || 'Chart'}</p>
                          <div className="chat-chart-wrap">
                            <ResponsiveContainer width="100%" height="100%">
                              {c.type === 'bar' ? (
                                <BarChart
                                  data={c.data || []}
                                  onClick={(state) => {
                                    const payload = state?.activePayload?.[0]?.payload || {};
                                    const xKey = c.x || 'x';
                                    const xVal = payload?.[xKey];
                                    if (isDateLike(xVal)) {
                                      openCalendarForPoint(String(xVal), c.title || 'taAI chart event', `Bar metric from taAI: ${JSON.stringify(payload)}`);
                                    }
                                  }}
                                >
                                  <CartesianGrid stroke="rgba(255,255,255,.08)" />
                                  <XAxis dataKey={c.x || 'x'} tick={{ fill: 'var(--muted)', fontSize: 10 }} />
                                  <YAxis tickFormatter={compact} tick={{ fill: 'var(--muted)', fontSize: 10 }} />
                                  <Tooltip formatter={(v) => Number.isFinite(Number(v)) ? compact(v) : v} />
                                  <Bar dataKey={c.ys?.[0] || 'y'} fill="#2ecdc4" radius={[4, 4, 0, 0]} />
                                </BarChart>
                              ) : (
                                <LineChart
                                  data={c.data || []}
                                  onClick={(state) => {
                                    const dateStr = state?.activeLabel;
                                    const payload = state?.activePayload?.[0]?.payload || {};
                                    if (isDateLike(dateStr)) {
                                      openCalendarForPoint(String(dateStr), c.title || 'taAI chart event', `Line metric from taAI: ${JSON.stringify(payload)}`);
                                    }
                                  }}
                                >
                                  <CartesianGrid stroke="rgba(255,255,255,.08)" />
                                  <XAxis dataKey={c.x || 'x'} tick={{ fill: 'var(--muted)', fontSize: 10 }} />
                                  <YAxis tickFormatter={compact} tick={{ fill: 'var(--muted)', fontSize: 10 }} />
                                  <Tooltip formatter={(v) => Number.isFinite(Number(v)) ? compact(v) : v} />
                                  {(c.ys || ['y']).map((yk, idx) => (
                                    <Line key={yk} type="monotone" dataKey={yk} dot={false} strokeWidth={2} stroke={idx === 0 ? '#2ecdc4' : '#7fb6ff'} />
                                  ))}
                                </LineChart>
                              )}
                            </ResponsiveContainer>
                          </div>
                        </CardContent>
                      </Card>
                    ))}
                  </div>
                )}
              </div>
            ))}
          </CardContent>

          <Separator />
          <CardContent className="session-row">
            {chatSessions.length === 0 ? <p className="field-label">No previous chats.</p> : chatSessions.map((s) => (
              <Button variant="ghost" className="session-btn" key={s.session_id} onClick={() => loadChatSession(s.session_id)}>{s.preview || s.session_id.slice(0, 8)}</Button>
            ))}
          </CardContent>
          <Separator />

          <CardContent className="chat-input-row">
            <Input value={chatText} onChange={(e) => setChatText(e.target.value)} placeholder="Ask taAI..." onKeyDown={(e) => e.key === 'Enter' && sendChat(chatText)} />
            <Button disabled={chatBusy} onClick={() => sendChat(chatText)}>{chatBusy ? '...' : 'Send'}</Button>
          </CardContent>
        </Card>

        <section className="chart-grid top">
          <Card className="chart-card wide">
            <CardHeader>
              <CardTitle>Weekly Sales vs Baseline vs Simulated</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="chart-wrap">
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart
                    data={trendData}
                    onClick={(state) => {
                      const point = state?.activePayload?.[0]?.payload;
                      if (point) openCalendarForTrendPoint(point);
                    }}
                  >
                    <CartesianGrid stroke="rgba(255,255,255,.08)" />
                    <XAxis dataKey="Date" tick={{ fill: 'var(--muted)', fontSize: 10 }} />
                    <YAxis tickFormatter={compact} tick={{ fill: 'var(--muted)', fontSize: 10 }} />
                    <Tooltip formatter={(v) => money(v)} />
                    <Line
                      type="monotone"
                      dataKey="Actual"
                      stroke="#16b6ad"
                      dot={(props) => (
                        <PointDot
                          {...props}
                          stroke="#16b6ad"
                          onPointClick={openCalendarForTrendPoint}
                        />
                      )}
                      strokeWidth={2.2}
                    />
                    <Line
                      type="monotone"
                      dataKey="Baseline"
                      stroke="#7fb6ff"
                      dot={(props) => <PointDot {...props} stroke="#7fb6ff" onPointClick={openCalendarForTrendPoint} />}
                      strokeWidth={2}
                    />
                    <Line
                      type="monotone"
                      dataKey="Simulated"
                      stroke="#f2cc72"
                      dot={(props) => <PointDot {...props} stroke="#f2cc72" onPointClick={openCalendarForTrendPoint} />}
                      strokeDasharray="5 4"
                      strokeWidth={2}
                    />
                  </LineChart>
                </ResponsiveContainer>
              </div>
              <p className="field-label">Point-click opens a Google Calendar task for that week.</p>
            </CardContent>
          </Card>

          <Card className="chart-card side">
            <CardHeader>
              <CardTitle>What-If Controls (% Change)</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="slider-grid">
                {FEATURES.map((f) => (
                  <div key={f}>
                    <label className="field-label">{f}: {adjustments[f]}%</label>
                    <input type="range" min="-30" max="30" value={adjustments[f]} onChange={(e) => setAdjustments((a) => ({ ...a, [f]: Number(e.target.value) }))} />
                  </div>
                ))}
              </div>

              <Card className="insight-card">
                <CardContent>
                  <p className="field-label">taAI diagnostic snapshot</p>
                  <p className="field-label">Avg actual: {money(insights?.snapshot?.avg_sales || 0)} | Avg predicted: {money(insights?.snapshot?.avg_pred || 0)}</p>
                  <p className="field-label">Residual mean: {compact(insights?.snapshot?.residual_mean || 0)} | Residual std: {compact(insights?.snapshot?.residual_std || 0)}</p>
                  <p className="field-label">Anomaly weeks: {Number(insights?.snapshot?.anomaly_count || 0)}</p>
                  <p className="field-label">Skewness: {Number(distStats?.skewness || 0).toFixed(3)} | Kurtosis: {Number(distStats?.kurtosis_pearson || 0).toFixed(3)}</p>
                  <p className="field-label">JB: {compact(distStats?.jarque_bera_stat || 0)} | p-value: {Number(distStats?.jarque_bera_pvalue || 0).toExponential(2)}</p>
                </CardContent>
              </Card>
            </CardContent>
          </Card>
        </section>

        <section className="chart-grid bottom">
          <Card className="chart-card wide">
            <CardHeader>
              <CardTitle>Scatter: Adjusted Factor vs Simulated Sales</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="chart-wrap">
                <ResponsiveContainer width="100%" height="100%">
                  <ScatterChart>
                    <CartesianGrid stroke="rgba(255,255,255,.08)" />
                    <XAxis type="number" domain={[scatterXMin, scatterXMax]} dataKey="x" tick={{ fill: 'var(--muted)', fontSize: 10 }} />
                    <YAxis type="number" domain={[scatterYMin, scatterYMax]} dataKey="y" tickFormatter={compact} tick={{ fill: 'var(--muted)', fontSize: 10 }} />
                    <Tooltip
                      formatter={(v, n) => (n === 'y' ? money(v) : Number(v).toFixed(3))}
                      labelFormatter={() => factor}
                      contentStyle={{ background: 'rgba(4, 23, 31, 0.95)', border: '1px solid rgba(176,210,214,.28)', borderRadius: 10 }}
                    />
                    <Scatter
                      data={scatterData.filter((d) => !d.holiday)}
                      name="Normal week"
                      fill="#2ecdc4"
                      onClick={(point) => {
                        if (point?.date) openCalendarForPoint(point.date, `taAI Scatter Review (${factor})`, `Normal-week point: factor=${point.x}, simulated=${point.y}`);
                      }}
                    />
                    <Scatter
                      data={scatterData.filter((d) => d.holiday)}
                      name="Holiday week"
                      fill="#f5c76e"
                      onClick={(point) => {
                        if (point?.date) openCalendarForPoint(point.date, `taAI Holiday Scatter Review (${factor})`, `Holiday-week point: factor=${point.x}, simulated=${point.y}`);
                      }}
                    />
                    <Line type="linear" data={scatterTrend} dataKey="y" stroke="#8ddcff" dot={false} strokeWidth={2} isAnimationActive={false} />
                  </ScatterChart>
                </ResponsiveContainer>
              </div>
              <p className="field-label">Trend: y = {scatterFit.slope.toFixed(2)}x + {scatterFit.intercept.toFixed(0)} | corr = {scatterFit.corr.toFixed(3)} | R² = {scatterFit.r2.toFixed(3)}</p>
            </CardContent>
          </Card>

          <Card className="chart-card side">
            <CardHeader>
              <CardTitle>Correlations + Coefficients</CardTitle>
            </CardHeader>
            <CardContent className="corr-grid">
              {corr.map((c) => (
                <div className="corr-item" key={c.feature}>
                  <div className="corr-top"><span>{c.feature}</span><span>{Number(c.corr).toFixed(3)}</span></div>
                  <div className="corr-track"><div className="corr-fill" style={{ width: `${Math.min(100, Math.abs(Number(c.corr)) * 100)}%`, background: Number(c.corr) >= 0 ? '#35c4a1' : '#f5b042' }} /></div>
                </div>
              ))}
            </CardContent>
          </Card>
        </section>
      </div>
    </div>
  );
}
