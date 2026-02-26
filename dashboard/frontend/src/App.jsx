import { useEffect, useMemo, useState } from 'react';
import {
  ResponsiveContainer,
  LineChart,
  Line,
  CartesianGrid,
  XAxis,
  YAxis,
  Tooltip,
  ScatterChart,
  Scatter,
} from 'recharts';
import './styles.css';

const FEATURES = ['CPI', 'Unemployment', 'Fuel_Price', 'Temperature'];
const money = (n) => new Intl.NumberFormat('en-US', { style: 'currency', currency: 'USD', maximumFractionDigits: 0 }).format(Number(n || 0));
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
  const [chatSuggestions, setChatSuggestions] = useState([]);
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

  async function loadChatSuggestions() {
    try {
      const res = await api('/api/taai/suggestions');
      setChatSuggestions(Array.isArray(res.suggestions) ? res.suggestions : []);
    } catch {
      setChatSuggestions([]);
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
    loadChatSuggestions();
    if (chatSessionId) loadChatSession(chatSessionId).catch(() => {});
  }, []);

  return (
    <div className="app">
      <div className="container">
        <div className="head">
          <div>
            <h1>Walmart Forecast Dashboard</h1>
            <div className="sub">Model-linked forecasting with interactive economist-grade diagnostics</div>
          </div>
          <div className="panel themeSwitch">
            <span className="sub">Theme</span>
            <button className={`themeBtn ${theme === 'dark' ? 'active' : ''}`} onClick={() => setTheme('dark')}>Dark</button>
            <button className={`themeBtn ${theme === 'light' ? 'active' : ''}`} onClick={() => setTheme('light')}>Light</button>
          </div>
        </div>

        <div className="grid grid12 topGrid">
          <div className="panel span8">
            <div className="row">
              <div>
                <label className="sub">Store</label>
                <select value={store} onChange={(e) => setStore(e.target.value)}>
                  <option value="all">All Stores</option>
                  {stores.map((s) => <option key={s.Store} value={String(s.Store)}>Store {s.Store}</option>)}
                </select>
              </div>
              <div>
                <label className="sub">Weeks: {weeks}</label>
                <input type="range" min="52" max="260" value={weeks} onChange={(e) => setWeeks(Number(e.target.value))} />
              </div>
            </div>
          </div>
          <div className="panel span4">
            <label className="sub">Scatter Factor</label>
            <select value={factor} onChange={(e) => setFactor(e.target.value)}>
              {FEATURES.map((f) => <option key={f} value={f}>{f}</option>)}
            </select>
            <div className="sub status">{status}</div>
          </div>
        </div>

        <div className="cards">
          <div className="card"><div className="k">Average Weekly Sales</div><div className="v">{money(overview?.avg_weekly_sales || 0)}</div></div>
          <div className="card"><div className="k">Peak Sales</div><div className="v">{money(overview?.peak_sales || 0)}</div></div>
          <div className="card"><div className="k">Holiday Avg</div><div className="v">{money(overview?.holiday_avg || 0)}</div></div>
          <div className="card"><div className="k">Holiday Count</div><div className="v">{overview?.holiday_count ?? 0}</div></div>
          <div className="card"><div className="k">Simulated Avg Sales</div><div className="v">{money(simulatedAvg)}</div></div>
        </div>

        <div className="panel inlineChat">
          <div className="chatHead">
            <div>
              <div className="chatTitle">taAI Economist Assistant</div>
              <div className="mini">Ask anything about sales, stores, model coefficients, creators, and dataset sources.</div>
            </div>
            <div className="chatHeadBtns">
              <button className="quickBtn" onClick={() => { setChatSessionId(''); localStorage.removeItem('taai_session_id'); setChatMessages([{ role: 'assistant', content: 'New chat started.' }]); }}>New</button>
            </div>
          </div>
          <div className="chatBody inlineBody">
            {chatMessages.map((m, i) => (
              <div key={i} className={`msg ${m.role === 'user' ? 'user' : 'bot'}`}>
                {m.content}
                {m.role !== 'user' && m.intent && (
                  <div className="msgMeta">
                    <span>{String(m.intent).replaceAll('_', ' ')}</span>
                    <span>{Math.round(Number(m.confidence || 0) * 100)}% confidence</span>
                  </div>
                )}
              </div>
            ))}
          </div>
          <div className="chatQuick">
            {(chatSuggestions.length ? chatSuggestions.slice(0, 8) : [
              'Who made taAI?',
              'Which dataset is this dashboard using?',
              'When was the highest sales week?',
              'When was the lowest sales week?',
              'Compare top 5 stores by average weekly sales',
              'Give me a 3-scenario forecast summary',
            ]).map((q) => (
              <button className="quickBtn" key={q} onClick={() => sendChat(q)}>{q}</button>
            ))}
          </div>
          <div className="chatSessions">
            {chatSessions.length === 0 ? <div className="mini">No previous chats.</div> : chatSessions.map((s) => (
              <button className="quickBtn" key={s.session_id} onClick={() => loadChatSession(s.session_id)}>{s.preview || s.session_id.slice(0, 8)}</button>
            ))}
          </div>
          <div className="chatInput">
            <input value={chatText} onChange={(e) => setChatText(e.target.value)} placeholder="Ask taAI..." onKeyDown={(e) => e.key === 'Enter' && sendChat(chatText)} />
            <button disabled={chatBusy} onClick={() => sendChat(chatText)}>{chatBusy ? '...' : 'Send'}</button>
          </div>
        </div>

        <div className="grid grid12 chartsTop">
          <div className="panel span8">
            <div className="title">Weekly Sales vs Baseline Prediction vs Simulated Prediction</div>
            <div className="chartWrap">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={trendData}>
                  <CartesianGrid stroke="rgba(255,255,255,.08)" />
                  <XAxis dataKey="Date" tick={{ fill: '#b7c8cd', fontSize: 10 }} />
                  <YAxis tickFormatter={compact} tick={{ fill: '#b7c8cd', fontSize: 10 }} />
                  <Tooltip formatter={(v) => money(v)} />
                  <Line type="monotone" dataKey="Actual" stroke="#16b6ad" dot={false} strokeWidth={2.2} />
                  <Line type="monotone" dataKey="Baseline" stroke="#7fb6ff" dot={false} strokeWidth={2.0} />
                  <Line type="monotone" dataKey="Simulated" stroke="#f2cc72" dot={false} strokeDasharray="5 4" strokeWidth={2.0} />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </div>

          <div className="panel span4">
            <div className="title">What-If Factor Controls (% Change)</div>
            <div className="sliderGrid">
              {FEATURES.map((f) => (
                <div key={f}>
                  <label className="mini">{f}: {adjustments[f]}%</label>
                  <input type="range" min="-30" max="30" value={adjustments[f]} onChange={(e) => setAdjustments((a) => ({ ...a, [f]: Number(e.target.value) }))} />
                </div>
              ))}
            </div>
            <div className="insightBox">
              <div className="mini">taAI insight snapshot (current filter)</div>
              <div className="mini">Avg actual: {money(insights?.snapshot?.avg_sales || 0)} | Avg predicted: {money(insights?.snapshot?.avg_pred || 0)}</div>
              <div className="mini">Residual mean: {compact(insights?.snapshot?.residual_mean || 0)} | Residual std: {compact(insights?.snapshot?.residual_std || 0)}</div>
              <div className="mini">Estimated anomaly weeks: {Number(insights?.snapshot?.anomaly_count || 0)}</div>
              <div className="mini">Skewness: {Number(distStats?.skewness || 0).toFixed(3)} | Kurtosis (Pearson): {Number(distStats?.kurtosis_pearson || 0).toFixed(3)}</div>
              <div className="mini">JB: {compact(distStats?.jarque_bera_stat || 0)} | p-value: {Number(distStats?.jarque_bera_pvalue || 0).toExponential(2)}</div>
            </div>
          </div>
        </div>

        <div className="grid grid12 chartsBottom">
          <div className="panel span8">
            <div className="title">Scatter: Adjusted Factor vs Simulated Sales</div>
            <div className="chartWrap">
              <ResponsiveContainer width="100%" height="100%">
                <ScatterChart>
                  <CartesianGrid stroke="rgba(255,255,255,.08)" />
                  <XAxis type="number" domain={[scatterXMin, scatterXMax]} dataKey="x" tick={{ fill: '#b7c8cd', fontSize: 10 }} />
                  <YAxis type="number" domain={[scatterYMin, scatterYMax]} dataKey="y" tickFormatter={compact} tick={{ fill: '#b7c8cd', fontSize: 10 }} />
                  <Tooltip
                    formatter={(v, n) => (n === 'y' ? money(v) : Number(v).toFixed(3))}
                    labelFormatter={() => factor}
                    contentStyle={{ background: 'rgba(4, 23, 31, 0.95)', border: '1px solid rgba(176,210,214,.28)', borderRadius: 10 }}
                  />
                  <Scatter data={scatterData.filter((d) => !d.holiday)} name="Normal week" fill="#2ecdc4" />
                  <Scatter data={scatterData.filter((d) => d.holiday)} name="Holiday week" fill="#f5c76e" />
                  <Line type="linear" data={scatterTrend} dataKey="y" stroke="#8ddcff" dot={false} strokeWidth={2} isAnimationActive={false} />
                </ScatterChart>
              </ResponsiveContainer>
            </div>
            <div className="mini scatterMeta">
              Trend: y = {scatterFit.slope.toFixed(2)}x + {scatterFit.intercept.toFixed(0)} | corr = {scatterFit.corr.toFixed(3)} | R² = {scatterFit.r2.toFixed(3)}
            </div>
          </div>

          <div className="panel span4">
            <div className="title">Correlations + Coefficients</div>
            <div className="bars">
              {corr.map((c) => (
                <div className="bar" key={c.feature}>
                  <div className="barTop"><span>{c.feature}</span><span>{Number(c.corr).toFixed(3)}</span></div>
                  <div className="barTrack"><div className="barFill" style={{ width: `${Math.min(100, Math.abs(Number(c.corr)) * 100)}%`, background: Number(c.corr) >= 0 ? '#35c4a1' : '#f5b042' }} /></div>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>

    </div>
  );
}
