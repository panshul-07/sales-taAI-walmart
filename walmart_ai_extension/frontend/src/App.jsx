import { useEffect, useState } from 'react';
import { api } from './api/client';
import Sidebar from './components/Sidebar';
import MessageBubble from './components/MessageBubble';
import ChartDisplay from './components/ChartDisplay';

export default function App() {
  const [sessionId, setSessionId] = useState('');
  const [sessions, setSessions] = useState([]);
  const [messages, setMessages] = useState([
    { role: 'assistant', content: 'taAI ready. Ask any Walmart sales question.' },
  ]);
  const [question, setQuestion] = useState('');
  const [loading, setLoading] = useState(false);
  const [charts, setCharts] = useState([]);
  const [table, setTable] = useState([]);
  const [suggestions, setSuggestions] = useState([]);

  async function sendMessage(text) {
    const msg = (text || '').trim();
    if (!msg || loading) return;

    setMessages((prev) => [...prev, { role: 'user', content: msg }]);
    setLoading(true);
    try {
      const res = await api.post('/chat', { message: msg, session_id: sessionId || null });
      const payload = res.data;
      setSessionId(payload.session_id);
      setMessages((prev) => [
        ...prev,
        { role: 'assistant', content: payload.answer, confidence: payload.confidence },
      ]);
      setCharts(payload.charts || []);
      setTable(payload.table || []);
      setSuggestions(payload.suggestions || []);
      await loadSessions();
    } catch (err) {
      setMessages((prev) => [...prev, { role: 'assistant', content: `Error: ${err.message}` }]);
    } finally {
      setLoading(false);
      setQuestion('');
    }
  }

  async function loadSessions() {
    try {
      const res = await api.get('/sessions');
      setSessions(res.data.sessions || []);
    } catch {
      setSessions([]);
    }
  }

  useEffect(() => {
    loadSessions();
  }, []);

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100">
      <div className="mx-auto flex max-w-7xl flex-col md:flex-row">
        <Sidebar
          sessions={sessions}
          onSelect={async (sid) => {
            setSessionId(sid);
            try {
              const res = await api.get(`/sessions/${sid}`);
              setMessages((res.data.messages || []).map((m) => ({ role: m.role, content: m.content })));
            } catch {
              // ignore and keep current messages
            }
          }}
          onNew={() => {
            setSessionId('');
            setMessages([{ role: 'assistant', content: 'New chat started.' }]);
            setCharts([]);
            setTable([]);
          }}
        />

        <main className="flex-1 p-4">
          <h1 className="mb-3 text-2xl font-bold">Walmart AI Sales Chatbot</h1>
          <div className="space-y-3">
            {messages.map((m, i) => (
              <MessageBubble key={i} message={m} />
            ))}
          </div>

          {charts.length > 0 && (
            <div className="mt-4 grid gap-3">
              {charts.map((c, i) => (
                <ChartDisplay key={i} chart={c} />
              ))}
            </div>
          )}

          {table.length > 0 && (
            <div className="mt-4 overflow-auto rounded-xl border border-slate-700">
              <table className="min-w-full text-sm">
                <thead className="bg-slate-900">
                  <tr>
                    {Object.keys(table[0]).map((k) => (
                      <th key={k} className="px-2 py-2 text-left">{k}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {table.slice(0, 100).map((row, idx) => (
                    <tr key={idx} className="border-t border-slate-800">
                      {Object.keys(row).map((k) => (
                        <td key={k} className="px-2 py-2">{String(row[k])}</td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}

          <div className="mt-4 flex gap-2">
            <input
              className="flex-1 rounded-lg border border-slate-700 bg-slate-900 px-3 py-2"
              value={question}
              onChange={(e) => setQuestion(e.target.value)}
              placeholder="Ask: Why did sales drop in March?"
              onKeyDown={(e) => e.key === 'Enter' && sendMessage(question)}
            />
            <button
              onClick={() => sendMessage(question)}
              disabled={loading}
              className="rounded-lg bg-cyan-500 px-4 py-2 font-semibold text-slate-950"
            >
              {loading ? 'Thinking...' : 'Send'}
            </button>
          </div>

          {suggestions.length > 0 && (
            <div className="mt-3 flex flex-wrap gap-2">
              {suggestions.map((s) => (
                <button key={s} onClick={() => sendMessage(s)} className="rounded-md border border-slate-700 px-2 py-1 text-xs">
                  {s}
                </button>
              ))}
            </div>
          )}
        </main>
      </div>
    </div>
  );
}
