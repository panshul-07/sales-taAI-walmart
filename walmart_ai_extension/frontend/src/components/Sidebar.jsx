export default function Sidebar({ sessions, onSelect, onNew }) {
  return (
    <aside className="w-full border-r border-slate-800 bg-slate-950 p-3 md:w-72">
      <button onClick={onNew} className="mb-3 w-full rounded-lg bg-cyan-500 px-3 py-2 font-semibold text-slate-950">
        New Chat
      </button>
      <div className="space-y-2">
        {sessions.length === 0 && <div className="text-sm text-slate-400">No sessions yet</div>}
        {sessions.map((s) => (
          <button
            key={s.session_id}
            onClick={() => onSelect(s.session_id)}
            className="w-full rounded-md border border-slate-700 px-2 py-2 text-left text-sm text-slate-200 hover:bg-slate-800"
          >
            {s.preview || s.session_id.slice(0, 10)}
          </button>
        ))}
      </div>
    </aside>
  );
}
