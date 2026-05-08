import React, { useState, useEffect, useRef } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
  Plus,
  Trash2,
  Edit3,
  Orbit,
  Compass,
  Activity,
  Paperclip,
  Send,
  Sparkles,
  FileText,
  X,
  Loader2,
  Square,
  Shield,
  Zap,
  BarChart3,
  Copy,
  Check
} from 'lucide-react';
import ReactMarkdown from 'react-markdown';
import remarkMath from 'remark-math';
import rehypeKatex from 'rehype-katex';
import 'katex/dist/katex.min.css';

const MarkdownRenderer = ({ content }) => {
  const [copiedCode, setCopiedCode] = useState(null);
  const copyCode = (code, idx) => {
    navigator.clipboard.writeText(code);
    setCopiedCode(idx);
    setTimeout(() => setCopiedCode(null), 2000);
  };
  return (
    <div className="prose-academic prose-invert max-w-none math-container">
      <ReactMarkdown
        remarkPlugins={[remarkMath]}
        rehypePlugins={[rehypeKatex]}
        components={{
          code({ node, inline, className, children, ...props }) {
            const match = /language-(\w+)/.exec(className || '');
            const codeStr = String(children).replace(/\n$/, '');
            if (!inline && (match || codeStr.includes('\n'))) {
              const idx = codeStr.slice(0, 30);
              return (
                <div className="relative group my-4 rounded-xl overflow-hidden border border-white/10">
                  <div className="flex items-center justify-between px-4 py-2 bg-white/5 border-b border-white/5">
                    <span className="text-[9px] font-black uppercase tracking-widest text-gray-500">{match ? match[1] : 'code'}</span>
                    <button onClick={() => copyCode(codeStr, idx)} className="flex items-center gap-1.5 text-[9px] font-bold text-gray-400 hover:text-white transition-colors">
                      {copiedCode === idx ? <><Check size={10} /> Copied</> : <><Copy size={10} /> Copy</>}
                    </button>
                  </div>
                  <pre className="p-4 overflow-x-auto bg-[#0d1b26] text-gray-300 text-sm leading-relaxed"><code {...props}>{children}</code></pre>
                </div>
              );
            }
            return <code className="px-1.5 py-0.5 bg-white/10 rounded text-accent-blue text-sm" {...props}>{children}</code>;
          }
        }}
      >
        {content}
      </ReactMarkdown>
    </div>
  );
};

const TypewriterText = ({ text, speed = 20, onComplete }) => {
  const [displayedText, setDisplayedText] = useState('');
  useEffect(() => {
    let i = 0;
    const interval = setInterval(() => {
      setDisplayedText(text.slice(0, i));
      i++;
      if (i > text.length) {
        clearInterval(interval);
        if (onComplete) onComplete();
      }
    }, speed);
    return () => clearInterval(interval);
  }, [text, speed, onComplete]);

  return <MarkdownRenderer content={displayedText} />;
};

const SkeletonLoader = () => (
  <div className="mt-6 space-y-4 animate-pulse-soft">
    <div className="flex items-center gap-3 mb-6">
      <div className="w-8 h-8 rounded-full bg-white/5 animate-shimmer" />
      <div className="h-4 bg-white/5 rounded-full w-48 animate-shimmer" />
    </div>
    <div className="space-y-3">
      <div className="h-3 bg-white/5 rounded-full w-full animate-shimmer" />
      <div className="h-3 bg-white/5 rounded-full w-11/12 animate-shimmer" />
      <div className="h-3 bg-white/5 rounded-full w-10/12 animate-shimmer" />
      <div className="h-3 bg-white/5 rounded-full w-9/12 animate-shimmer" />
    </div>
  </div>
);

const ThinkingStatus = () => {
  const [step, setStep] = useState(0);
  const steps = [
    "Analyzing course materials...",
    "Extracting academic concepts...",
    "Identifying historical exam trends...",
    "Synthesizing high-probability questions...",
    "Running cross-reference analysis..."
  ];

  useEffect(() => {
    const interval = setInterval(() => setStep(s => (s + 1) % steps.length), 3000);
    return () => clearInterval(interval);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  return (
    <div className="flex items-center gap-3 text-[#1B9AAA] font-black uppercase tracking-[0.2em] text-[10px] mb-6">
      <Loader2 size={14} className="animate-spin" />
      <span>{steps[step]}</span>
    </div>
  );
};

const CircularProgress = ({ progress }) => {
  const radius = 10;
  const circumference = 2 * Math.PI * radius;
  return (
    <svg width="24" height="24" className="rotate-[-90deg]">
      <circle cx="12" cy="12" r={radius} fill="transparent" stroke="rgba(255,255,255,0.1)" strokeWidth="2" />
      <circle cx="12" cy="12" r={radius} fill="transparent" stroke="#1B9AAA" strokeWidth="2" strokeDasharray={circumference} strokeDashoffset={circumference - (progress / 100) * circumference} strokeLinecap="round" className="transition-all duration-300" />
    </svg>
  );
};

const CopyButton = ({ data }) => {
  const [copied, setCopied] = useState(false);
  const handleCopy = () => {
    const text = [
      `Topic: ${data.predicted_topic}`,
      `Confidence: ${data.overall_confidence}`,
      `Engine: ${data.engine}`,
      `Hot Topics: ${data.hot_topics?.join(', ')}`,
      '',
      ...(data.questions?.map((q, i) => `Q${q.id} (${q.probability}):\n${q.question}`) || [])
    ].join('\n\n');
    navigator.clipboard.writeText(text);
    setCopied(true);
    setTimeout(() => setCopied(false), 2500);
  };
  return (
    <button onClick={handleCopy} className="flex items-center gap-2 px-4 py-2 rounded-xl text-[10px] font-black uppercase tracking-widest text-gray-400 hover:text-white bg-white/5 hover:bg-white/10 border border-white/5 hover:border-white/10 transition-all active:scale-95">
      {copied ? <><Check size={13} /> Copied to clipboard</> : <><Copy size={13} /> Copy response</>}
    </button>
  );
};

const ResultsBlock = ({ data, onSubtaskLoading }) => {
  const [revealed, setRevealed] = useState({});
  const [loadingAnswer, setLoadingAnswer] = useState({});
  const [isTyping, setIsTyping] = useState({});
  const abortControllerRef = useRef(null);

  const fetchAnswer = async (id, question) => {
    abortControllerRef.current = new AbortController();
    setLoadingAnswer(p => ({ ...p, [id]: true }));
    onSubtaskLoading(true);
    try {
      const res = await fetch("/api/generate-answer", {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question, context: data?.retrieved_context_used || "" }),
        signal: abortControllerRef.current.signal
      });
      const resData = await res.json();
      if (resData.success) {
        setIsTyping(p => ({ ...p, [id]: true }));
        setRevealed(p => ({ ...p, [id]: resData.answer }));
      }
    } catch (e) { if (e.name !== 'AbortError') console.error(e); }
    finally { setLoadingAnswer(p => ({ ...p, [id]: false })); onSubtaskLoading(false); }
  };

  const stopGeneration = () => { if (abortControllerRef.current) abortControllerRef.current.abort(); };

  if (!data) return <div className="p-6 bg-rose-500/10 border border-rose-500/20 rounded-3xl text-rose-400 font-bold uppercase text-xs">⚠️ Generation returned empty data. Please try again.</div>;

  return (
    <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="mt-6 space-y-12">
      <div className="space-y-6">
        <div className="flex items-center gap-3">
          <Orbit size={16} className="text-accent-blue opacity-50" />
          <h2 className="text-[10px] font-black uppercase tracking-[0.3em] text-gray-400">Core Academic Trends</h2>
        </div>
        <div className="flex flex-wrap gap-2">
          {data.hot_topics?.map((topic, i) => (
            <span key={i} className="premium-tag premium-tag-cyan px-4 py-1.5 rounded-full text-[10px] font-black uppercase tracking-widest flex items-center gap-2 hover:scale-105 transition-transform">
              <Zap size={10} /> {topic}
            </span>
          ))}
        </div>
      </div>

      {data.questions?.length > 0 && (
        <div className="space-y-16">
          <div className="flex items-center justify-between border-b border-white/5 pb-4">
            <div className="flex items-center gap-3">
              <Sparkles size={16} className="text-accent-wood opacity-60" />
              <h2 className="text-[10px] font-black uppercase tracking-[0.3em] text-gray-400">Predicted Exam Matrix</h2>
            </div>
            <div className="premium-tag px-4 py-1.5 rounded-full border border-white/10 flex items-center gap-3">
              <Shield size={12} className="text-accent-blue" />
              <div className="flex flex-col">
                <span className="text-[7px] font-black text-gray-500 uppercase tracking-widest leading-none">AI Confidence</span>
                <span className="text-[10px] font-black text-accent-blue leading-none">{data.overall_confidence}</span>
              </div>
            </div>
          </div>

          <div className="space-y-20">
            {data.questions?.map((q, i) => (
              <div key={i} className="relative">
                <div className="flex items-center gap-4 mb-6">
                  <span className="text-[10px] font-black text-white/10 uppercase tracking-[0.4em]">Section {q.id}</span>
                  <div className="h-px flex-1 bg-white/5" />
                  <span className="premium-tag premium-tag-cyan px-4 py-1.5 rounded-full text-[9px] font-black uppercase tracking-widest flex items-center gap-2">
                    <BarChart3 size={11} /> Likelihood: {q.probability}
                  </span>
                </div>

                <div className="text-[1.1rem] text-gray-100 leading-relaxed math-container pl-2 border-l-2 border-accent-blue/20">
                  <MarkdownRenderer content={q.question} />
                </div>

                <div className="mt-8 pl-6">
                  {revealed[q.id] ? (
                    <motion.div initial={{ opacity: 0, x: -10 }} animate={{ opacity: 1, x: 0 }} className="relative group">
                      <div className="absolute -left-6 top-0 bottom-0 w-0.5 bg-accent-blue/40" />
                      <div className="text-[9px] text-accent-blue/70 font-black uppercase tracking-[0.2em] mb-4">Official Prediction Solution</div>
                      <div className="text-gray-300 text-[1rem] leading-relaxed">
                        {isTyping[q.id] ? <TypewriterText text={revealed[q.id]} speed={12} onComplete={() => setIsTyping(p => ({ ...p, [q.id]: false }))} /> : <MarkdownRenderer content={revealed[q.id]} />}
                      </div>
                    </motion.div>
                  ) : (
                    <div className="flex items-center gap-4">
                      <button onClick={() => fetchAnswer(q.id, q.question)} disabled={loadingAnswer[q.id]} className="flex items-center gap-3 px-6 py-2.5 bg-accent-blue/10 hover:bg-accent-blue/20 border border-accent-blue/20 rounded-xl text-[10px] font-black uppercase tracking-widest text-accent-blue transition-all active:scale-95 disabled:opacity-50">
                        {loadingAnswer[q.id] ? <Loader2 size={14} className="animate-spin" /> : <Sparkles size={14} />} Reveal Solution
                      </button>
                      {loadingAnswer[q.id] && <button onClick={stopGeneration} className="p-2.5 bg-rose-500/10 text-rose-400 border border-rose-500/20 rounded-xl hover:bg-rose-500/20 transition-all flex items-center gap-2 text-[9px] font-black uppercase"><Square size={12} fill="currentColor" /> Stop</button>}
                    </div>
                  )}
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Engine info + Copy button */}
      <div className="flex items-center justify-between border-t border-white/5 pt-6">
        <div className="text-[9px] font-bold text-gray-500 uppercase tracking-widest">
          Engine: {data.engine}
        </div>
        <CopyButton data={data} />
      </div>
    </motion.div>
  );
};

function App() {
  const [sessions, setSessions] = useState(() => {
    const saved = localStorage.getItem('exam_sessions_v2');
    return saved ? JSON.parse(saved) : [{ id: Date.now(), title: 'New chat', history: [], contextNotes: [] }];
  });
  const [activeSessionId, setActiveSessionId] = useState(sessions[0].id);
  const [stagedFiles, setStagedFiles] = useState([]);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [isUploading, setIsUploading] = useState(false);
  const [query, setQuery] = useState('');
  const [loading, setLoading] = useState(false);
  const [subtaskLoading, setSubtaskLoading] = useState(false);
  const [modelChoice, setModelChoice] = useState('option1');
  const [editingSessionId, setEditingSessionId] = useState(null);
  const [tempTitle, setTempTitle] = useState('');
  const abortControllerRef = useRef(null);
  const fileInputRef = useRef(null);
  const chatEndRef = useRef(null);

  const activeSession = sessions.find(s => s.id === activeSessionId) || sessions[0];

  useEffect(() => {
    localStorage.setItem('exam_sessions_v2', JSON.stringify(sessions));
    chatEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [sessions, loading]);

  const createNewChat = () => {
    const newSession = { id: Date.now(), title: 'New chat', history: [], contextNotes: [] };
    setSessions([newSession, ...sessions]);
    setActiveSessionId(newSession.id);
  };

  const updateSessionTitle = (id, title) => setSessions(prev => prev.map(s => s.id === id ? { ...s, title: title.length > 25 ? title.slice(0, 25) + '...' : title } : s));

  const deleteSession = (id, e) => {
    e.stopPropagation();
    const newSessions = sessions.filter(s => s.id !== id);
    if (newSessions.length === 0) {
      const fresh = { id: Date.now(), title: 'New chat', history: [], contextNotes: [] };
      setSessions([fresh]);
      setActiveSessionId(fresh.id);
    } else {
      setSessions(newSessions);
      if (activeSessionId === id) setActiveSessionId(newSessions[0].id);
    }
  };

  const startEditing = (id, title, e) => {
    e.stopPropagation();
    setEditingSessionId(id);
    setTempTitle(title);
  };

  const saveTitle = (id) => {
    updateSessionTitle(id, tempTitle);
    setEditingSessionId(null);
  };

  const handleFileUpload = (e) => {
    if (e.target.files) setStagedFiles([...stagedFiles, ...Array.from(e.target.files)]);
  };

  const removeFile = (name) => {
    setStagedFiles(stagedFiles.filter(f => f.name !== name));
  };

  const stopMainGeneration = () => {
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
      setLoading(false);
      setIsUploading(false);
      setUploadProgress(0);
    }
  };

  const executeTask = async (mode = 'predict') => {
    abortControllerRef.current = new AbortController();
    const files = [...stagedFiles];
    const userQuery = query;
    if (files.length === 0 && !userQuery && activeSession.contextNotes.length === 0) return;

    setStagedFiles([]);
    setQuery('');
    setLoading(true);

    const userMsg = { role: 'user', content: userQuery || (mode === 'predict' ? 'Predict exam trends from uploaded material.' : 'Identify academic domain.'), files: files.map(f => f.name) };
    updateActiveHistory(userMsg);

    try {
      let currentNotes = activeSession.contextNotes;
      if (files.length > 0) {
        setIsUploading(true);
        const formData = new FormData();
        files.forEach(f => formData.append('files', f));

        const resData = await new Promise((resolve, reject) => {
          const xhr = new XMLHttpRequest();
          xhr.open("POST", "/api/upload-notes");
          xhr.upload.onprogress = (e) => {
            if (e.lengthComputable) {
              const pct = (e.loaded / e.total) * 100;
              setUploadProgress(pct);
            }
          };
          // Clear upload badge as soon as bytes are fully sent
          xhr.upload.onload = () => {
            setIsUploading(false);
            setUploadProgress(0);
          };
          xhr.onload = () => {
            if (xhr.status >= 200 && xhr.status < 300) {
              resolve(JSON.parse(xhr.responseText));
            } else {
              reject(new Error(`Upload failed with status ${xhr.status}: ${xhr.statusText}`));
            }
          };
          xhr.onerror = () => reject(new Error("Upload failed: Network error"));
          abortControllerRef.current.signal.addEventListener('abort', () => xhr.abort());
          xhr.send(formData);
        });

        currentNotes = resData.extracted_notes;
        setSessions(prev => prev.map(s => s.id === activeSessionId ? { ...s, contextNotes: currentNotes } : s));
      }

      const endpoints = { 'predict': '/api/predict-exam', 'topic': '/api/classify-exam' };
      const res = await fetch(`${endpoints[mode]}`, {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ notes: currentNotes, model_choice: modelChoice }),
        signal: abortControllerRef.current.signal
      });
      const data = await res.json();
      updateActiveHistory({ role: 'ai', data: data?.data });
      if (activeSession.title === 'New chat' && data?.data?.predicted_topic) {
        updateSessionTitle(activeSession.id, data.data.predicted_topic);
      }
    } catch (e) {
      if (e.name !== 'AbortError') updateActiveHistory({ role: 'ai', content: `❌ Error: ${e.message}` });
    } finally {
      setLoading(false);
      setIsUploading(false);
      setUploadProgress(0);
    }
  };

  const updateActiveHistory = (newMsg) => setSessions(prev => prev.map(s => s.id === activeSessionId ? { ...s, history: [...s.history, newMsg] } : s));

  const isInputValid = query.trim().length > 0 || stagedFiles.length > 0 || activeSession.contextNotes.length > 0;
  const isGlobalBusy = loading || subtaskLoading;

  return (
    <div className="flex h-screen bg-bg-main text-white selection:bg-accent-blue/30 overflow-hidden font-sans">
      {/* Sidebar */}
      <aside className="w-72 bg-bg-sidebar border-r border-white/5 flex flex-col transition-all duration-500 relative z-20 shadow-[10px_0_30px_rgba(0,0,0,0.2)]">
        <div className="p-6">
          <button onClick={createNewChat} disabled={isGlobalBusy} className="w-full group relative overflow-hidden px-6 py-4 bg-white/5 hover:bg-white/10 border border-white/10 rounded-2xl flex items-center gap-4 transition-all duration-300 disabled:opacity-50">
            <div className="absolute inset-0 bg-gradient-to-r from-accent-blue/0 via-accent-blue/5 to-accent-blue/0 translate-x-[-100%] group-hover:translate-x-[100%] transition-transform duration-1000" />
            <Plus size={18} className="text-accent-blue group-hover:scale-110 transition-transform" />
            <span className="text-[11px] font-black uppercase tracking-[0.2em] text-gray-300 group-hover:text-white">New Chat</span>
          </button>
        </div>

        <div className="flex-1 overflow-y-auto scrollbar-gemini px-4 space-y-2 pb-6">
          <div className="px-4 py-2 text-[9px] font-black uppercase tracking-[0.3em] text-gray-500 mb-2">History</div>
          <AnimatePresence mode="popLayout">
            {sessions.length === 0 ? (
              <div className="p-8 text-center space-y-3">
                <div className="w-12 h-12 bg-white/5 rounded-full mx-auto flex items-center justify-center opacity-20"><Activity size={24} /></div>
                <p className="text-[10px] font-bold text-gray-600 uppercase tracking-widest">No active sessions</p>
              </div>
            ) : sessions.map(s => (
              <motion.div key={s.id} layout initial={{ opacity: 0, x: -20 }} animate={{ opacity: 1, x: 0 }} exit={{ opacity: 0, scale: 0.95 }} className="group relative">
                <div onClick={() => !isGlobalBusy && setActiveSessionId(s.id)} className={`w-full text-left px-5 py-4 rounded-2xl flex items-center justify-between transition-all duration-300 cursor-pointer ${activeSessionId === s.id ? 'bg-accent-blue/10 border border-accent-blue/20 text-accent-blue' : 'hover:bg-white/5 border border-transparent text-gray-400 hover:text-gray-200'}`}>
                  <div className="flex items-center gap-4 truncate">
                    <Orbit size={14} className={activeSessionId === s.id ? 'text-accent-blue' : 'text-gray-500'} />
                    {editingSessionId === s.id ? (
                      <input autoFocus value={tempTitle} onChange={e => setTempTitle(e.target.value)} onBlur={() => saveTitle(s.id)} onKeyDown={(e) => e.key === 'Enter' && saveTitle(s.id)} onClick={e => e.stopPropagation()} className="bg-transparent border-none outline-none text-white w-full text-[11px] font-bold" />
                    ) : (
                      <span className="text-[11px] font-bold truncate tracking-wide">{s.title}</span>
                    )}
                  </div>
                  <div className="flex items-center gap-0.5 opacity-0 group-hover:opacity-100 transition-opacity">
                    <button onClick={(e) => { e.stopPropagation(); startEditing(s.id, s.title, e); }} className="p-1.5 hover:bg-white/10 rounded-lg text-gray-500 hover:text-white transition-all"><Edit3 size={11} /></button>
                    <button onClick={(e) => { e.stopPropagation(); deleteSession(s.id, e); }} className="p-1.5 hover:bg-rose-500/10 rounded-lg text-gray-500 hover:text-rose-400 transition-all"><Trash2 size={11} /></button>
                  </div>
                </div>
              </motion.div>
            ))}
          </AnimatePresence>
        </div>
      </aside>

      {/* Main Area */}
      <main className="flex-1 flex flex-col relative overflow-hidden bg-gradient-to-br from-bg-main via-[#162a3a] to-[#11212F]">
        <header className="h-20 border-b border-white/5 flex items-center justify-between px-10 bg-bg-main/80 backdrop-blur-3xl z-10">
          <div className="flex items-center gap-4">
            <div className="flex items-center gap-2 px-3 py-1.5 bg-accent-blue/10 rounded-full border border-accent-blue/20 animate-glow-pulse">
              <div className="w-1.5 h-1.5 rounded-full bg-accent-blue animate-pulse" />
              <span className="text-[9px] font-black uppercase tracking-widest text-accent-blue">{activeSession?.title || 'System Active'}</span>
            </div>
          </div>
          <div className="flex bg-white/5 p-1 rounded-2xl border border-white/10">
            {['option1', 'option2'].map(m => (
              <button key={m} onClick={() => setModelChoice(m)} disabled={isGlobalBusy} className={`px-6 py-2 rounded-xl text-[10px] font-black uppercase tracking-widest transition-all ${modelChoice === m ? 'bg-accent-blue text-white shadow-[0_0_20px_rgba(27,154,170,0.4)] scale-[1.02]' : 'text-gray-400 hover:text-white'}`}>{m === 'option1' ? 'QWEN-2.5' : 'SCIQ-NET'}</button>
            ))}
          </div>
        </header>

        <div className="flex-1 overflow-y-auto scrollbar-gemini relative">
          <div className="max-w-5xl mx-auto px-6 py-20">
            <AnimatePresence initial={false}>
              {activeSession?.history.length === 0 && (
                <motion.div initial={{ opacity: 0, y: 30 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.8, ease: 'easeOut' }} className="text-center py-32">
                  <motion.h1 animate={{ y: [0, -6, 0] }} transition={{ duration: 5, repeat: Infinity, ease: 'easeInOut' }} className="text-6xl font-black mb-8 gradient-text pb-2 tracking-tight">Academic Oracle</motion.h1>
                  <p className="text-gray-400 text-xl max-w-xl mx-auto font-medium">Deep predictive analysis for your academic journey.</p>
                  <div className="flex justify-center gap-3 mt-8">
                    <div className="w-12 h-1 rounded-full bg-accent-blue/40" />
                    <div className="w-8 h-1 rounded-full bg-accent-wood/40" />
                    <div className="w-4 h-1 rounded-full bg-accent-espresso/40" />
                  </div>
                </motion.div>
              )}
              {activeSession?.history.map((msg, i) => (
                <motion.div key={i} initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} className={`mb-12 flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}>
                  <div className={`max-w-[85%] ${msg.role === 'user' ? 'bg-accent-blue/10 border border-accent-blue/20 px-6 py-4 rounded-3xl rounded-tr-none' : 'w-full'}`}>
                    {msg.role === 'ai' ? (
                      msg.data ? <ResultsBlock data={msg.data} onSubtaskLoading={setSubtaskLoading} /> : <MarkdownRenderer content={msg.content} />
                    ) : (
                      <div className="space-y-3">
                        {msg.files && msg.files.length > 0 && (
                          <div className="flex flex-wrap gap-2 pb-2">
                            {msg.files.map((fname, fi) => (
                              <div key={fi} className="flex items-center gap-2 px-3 py-1 bg-accent-blue/10 border border-accent-blue/20 rounded-xl text-[10px] font-black uppercase tracking-widest text-accent-blue">
                                <FileText size={12} />
                                {fname}
                              </div>
                            ))}
                          </div>
                        )}
                        <p className="text-gray-100 text-[1rem] leading-relaxed font-medium">{msg.content}</p>
                      </div>
                    )}
                  </div>
                </motion.div>
              ))}
            </AnimatePresence>
            {loading && (
              <div className="space-y-4">
                <ThinkingStatus />
                <SkeletonLoader />
              </div>
            )}
            <div ref={chatEndRef} />
          </div>
        </div>

        <div className="absolute bottom-0 left-0 right-0 p-6 bg-gradient-to-t from-bg-main via-bg-main to-transparent z-20">
          <div className="max-w-7xl mx-auto">
            <div className="bg-bg-sidebar border border-white/10 rounded-[28px] p-4 shadow-[0_20px_50px_rgba(0,0,0,0.5)] focus-within:border-accent-blue/40 transition-all">
              <div className="flex flex-wrap gap-2 mb-3 px-2">
                {stagedFiles.map((f, i) => (
                  <div key={i} className="flex items-center gap-2 px-3 py-1 bg-accent-blue/10 text-accent-blue rounded-xl text-[10px] font-black uppercase border border-accent-blue/20">
                    <FileText size={12} /> {f.name}
                    <button onClick={() => removeFile(f.name)} className="hover:text-white transition-colors"><X size={12} /></button>
                  </div>
                ))}
                {isUploading && (
                  <div className="flex items-center gap-2 px-3 py-1 bg-white/5 rounded-xl border border-white/10">
                    <CircularProgress progress={uploadProgress} />
                    <span className="text-[9px] font-black uppercase text-gray-400">
                      {uploadProgress >= 100 ? 'Analyzing...' : `Uploading (${Math.round(uploadProgress)}%)...`}
                    </span>
                  </div>
                )}
              </div>

              <div className="flex items-center gap-4 px-2">
                <button onClick={() => fileInputRef.current.click()} disabled={isGlobalBusy} className="p-2.5 bg-white/5 hover:bg-white/10 border border-white/5 rounded-xl cursor-pointer text-gray-400 hover:text-white transition-all active:scale-95"><input type="file" ref={fileInputRef} multiple accept=".pdf" className="hidden" onChange={handleFileUpload} /><Paperclip size={20} /></button>
                <textarea value={query} onChange={(e) => !isGlobalBusy && setQuery(e.target.value)} onKeyDown={(e) => e.key === 'Enter' && !e.shiftKey && !isGlobalBusy && (e.preventDefault(), executeTask())} disabled={isGlobalBusy} placeholder={isGlobalBusy ? "Generating response..." : "Ask or upload lecture notes..."} className={`flex-1 bg-transparent border-none outline-none text-white text-[0.95rem] placeholder:text-gray-400 resize-none max-h-32 py-1 scrollbar-hide font-medium transition-opacity ${isGlobalBusy ? 'opacity-40 cursor-not-allowed' : ''}`} rows={1} />

                {isGlobalBusy ? (
                  <button onClick={stopMainGeneration} className="p-3 rounded-xl bg-rose-500/20 text-rose-400 border border-rose-500/30 hover:bg-rose-500/30 transition-all active:scale-95 flex items-center justify-center shadow-[0_0_20px_rgba(244,63,94,0.2)]">
                    <Square size={20} fill="currentColor" />
                  </button>
                ) : (
                  <button onClick={() => executeTask()} disabled={!isInputValid} className={`p-3 rounded-xl transition-all ${isInputValid ? 'bg-accent-blue text-white shadow-[0_0_20px_rgba(27,154,170,0.4)] hover:brightness-110 active:scale-95' : 'text-gray-600 bg-white/5'}`}>
                    <Send size={20} />
                  </button>
                )}
              </div>

              <div className="flex flex-wrap gap-2 mt-4 px-2 border-t border-white/5 pt-3">
                {[
                  { id: 'predict', icon: Activity, label: 'Predict Trends', color: 'bg-accent-blue/15 text-accent-blue border-accent-blue/30 hover:bg-accent-blue/25' },
                  { id: 'topic', icon: Compass, label: 'Identify Topic', color: 'bg-accent-wood/15 text-accent-wood border-accent-wood/30 hover:bg-accent-wood/25' }
                ].map(action => (
                  <button key={action.id} onClick={() => executeTask(action.id)} disabled={isGlobalBusy || !isInputValid} className={`flex items-center gap-2.5 px-6 py-2.5 rounded-xl text-[10px] font-black uppercase tracking-wider border transition-all duration-300 disabled:opacity-20 active:scale-95 ${action.color}`}>
                    <action.icon size={14} /> {action.label}
                  </button>
                ))}
              </div>
            </div>
          </div>
        </div>
      </main>
    </div>
  );
}

export default App;
