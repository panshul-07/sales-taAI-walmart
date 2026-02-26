export default function MessageBubble({ message }) {
  const isUser = message.role === 'user';
  return (
    <div className={`max-w-3xl rounded-xl border px-3 py-2 ${isUser ? 'ml-auto border-cyan-500/40 bg-cyan-900/20' : 'mr-auto border-slate-700 bg-slate-900'}`}>
      <div className="whitespace-pre-wrap text-sm text-slate-100">{message.content}</div>
      {!isUser && message.confidence !== undefined && (
        <div className="mt-1 text-xs text-slate-400">confidence: {Math.round(message.confidence * 100)}%</div>
      )}
    </div>
  );
}
