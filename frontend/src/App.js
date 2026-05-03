import React, { useState, useRef, useEffect } from 'react';
import ReactMarkdown from 'react-markdown';
import remarkMath from 'remark-math';
import rehypeKatex from 'rehype-katex';
import 'katex/dist/katex.min.css';
import './App.css';

const TypewriterText = ({ text, delay = 8, startTyping = true, onComplete }) => {
  const [currentText, setCurrentText] = useState('');

  useEffect(() => {
    if (!startTyping) return;
    
    if (currentText.length < text.length) {
      const timeout = setTimeout(() => {
        // Randomly chunk characters to mathematically simulate real network streaming variances
        const charsToAdd = Math.floor(Math.random() * 4) + 1;
        setCurrentText(text.slice(0, currentText.length + charsToAdd));
      }, delay);
      
      return () => clearTimeout(timeout);
    } else if (onComplete) {
      onComplete();
    }
  }, [currentText, text, startTyping, delay, onComplete]);

  return (
    <>
      <ReactMarkdown remarkPlugins={[remarkMath]} rehypePlugins={[rehypeKatex]}>
        {currentText + (currentText.length < text.length ? '▍' : '')}
      </ReactMarkdown>
    </>
  );
};

const MLResultsBlock = ({ msg }) => {
  const data = msg.data;
  const [completedIndex, setCompletedIndex] = useState(-1);

  return (
    <div className="ml-results-block" style={{ width: '100%' }}>
      <div className="result-header" style={{ display: 'flex', flexDirection: 'column', gap: '15px', width: '100%' }}>
        <div style={{ display: 'flex', flexWrap: 'wrap', gap: '10px' }}>
          <span className="topic-badge">🎯 Core Target Subject: {data.predicted_topic}</span>
        </div>
        
        {data.hot_topics && data.hot_topics.length > 0 && (
          <div className="hot-topics-container" style={{ background: 'rgba(239, 68, 68, 0.1)', padding: '12px', borderRadius: '8px', borderLeft: '4px solid #ef4444' }}>
            <h4 style={{ margin: '0 0 10px 0', color: '#fca5a5', fontSize: '0.9rem', textTransform: 'uppercase', letterSpacing: '1px' }}>
              🔥 High Probability Exam Topics
            </h4>
            <div style={{ display: 'flex', flexWrap: 'wrap', gap: '8px' }}>
              {data.hot_topics.map((ht, idx) => (
                <span key={idx} style={{ background: 'rgba(239, 68, 68, 0.2)', padding: '5px 12px', borderRadius: '20px', fontSize: '0.85rem', color: '#fecaca', whiteSpace: 'nowrap', border: '1px solid rgba(239, 68, 68, 0.3)' }}>
                  #{idx + 1} {ht}
                </span>
              ))}
            </div>
          </div>
        )}
      </div>
      
      <h4 style={{ marginTop: '20px' }}>5 Predicted Target Exam Queries</h4>
      <div style={{ display: 'flex', flexDirection: 'column', gap: '15px', marginTop: '10px' }}>
        {data.questions && data.questions.map((q, idx) => {
          const isTypingAllowed = completedIndex >= idx - 1;
          
          return isTypingAllowed ? (
            <div key={q.id} className="result-text" style={{ padding: '1.2rem', animation: 'fadeIn 0.5s ease-in-out' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '10px', borderBottom: '1px solid rgba(255,255,255,0.1)', paddingBottom: '8px' }}>
                <span style={{ fontWeight: 'bold', color: 'var(--accent-blue)' }}>Mock Output #{q.id}</span>
                <span className="conf-badge" style={{ background: 'rgba(16, 185, 129, 0.15)', color: '#34d399', border: '1px solid rgba(16, 185, 129, 0.4)' }}>
                  📊 Probability Link: {q.probability}
                </span>
              </div>
              <div style={{ whiteSpace: 'normal', lineHeight: '1.6', fontSize: '1.05rem', overflowX: 'auto', background: 'rgba(0,0,0,0.1)', padding: '10px', borderRadius: '8px' }}>
                <TypewriterText 
                  text={q.question.replace(/\\\(/g, '$').replace(/\\\)/g, '$').replace(/\\\[/g, '$$$$').replace(/\\\]/g, '$$$$')}
                  startTyping={isTypingAllowed}
                  onComplete={() => setCompletedIndex(idx)}
                  delay={5}
                />
              </div>
            </div>
          ) : null;
        })}
      </div>

      <details className="context-expander" style={{ marginTop: '2rem' }}>
        <summary>Audit Target FAISS Embedding Tensors</summary>
        <div className="context-text">{data.retrieved_context_used}</div>
      </details>
    </div>
  );
};

function App() {
  const [chatHistory, setChatHistory] = useState([
    {
      role: 'assistant',
      content: 'Hello! I am the AI Exam Predictor Copilot. Upload up to 10 course PDFs securely, and I will analyze them using Deep Text Extraction and FAISS semantic modeling to dynamically generate creative and challenging mock exam questions on theoretical topics.'
    }
  ]);
  const [stagedFiles, setStagedFiles] = useState([]);
  const [loading, setLoading] = useState(false);
  const fileInputRef = useRef(null);
  const chatEndRef = useRef(null);

  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [chatHistory, loading]);

  const handleFileChange = (e) => {
    if (e.target.files && e.target.files.length > 0) {
      validateAndSetFiles([...stagedFiles, ...Array.from(e.target.files)]);
    }
  };

  const validateAndSetFiles = (selectedFiles) => {
    if (selectedFiles.length > 10) {
      alert("Pipeline Error: Maximum of 10 document vectors are supported per pipeline request.");
      return;
    }
    const totalSize = selectedFiles.reduce((acc, file) => acc + file.size, 0);
    if (totalSize > 50 * 1024 * 1024) {
      alert("Pipeline Error: Cumulative byte constraints exceeded (Maximum 50 Megabytes).");
      return;
    }
    setStagedFiles(selectedFiles);
  };

  const removeStagedFile = (indexToRemove) => {
    setStagedFiles(stagedFiles.filter((_, idx) => idx !== indexToRemove));
  };

  const handleSend = async () => {
    if (stagedFiles.length === 0) return;
    
    const filesToProcess = [...stagedFiles];
    setStagedFiles([]);
    setLoading(true);

    setChatHistory(prev => [...prev, {
      role: 'user',
      content: `Analyze the contents of these documents over the PyTorch backend and generate a max-scale predictive exam protocol.`,
      files: filesToProcess.map(f => f.name)
    }]);

    try {
      const formData = new FormData();
      filesToProcess.forEach(file => formData.append('files', file));
      
      const uploadRes = await fetch("http://localhost:8000/api/upload-notes", {
        method: "POST", body: formData
      });
      const uploadData = await uploadRes.json();
      if (!uploadRes.ok) throw new Error(uploadData.detail || "PDF Extractor Crash");
      
      const dummyHistories = { "Backend Topic Flag Ignore": { 2024: 1 } };

      const predictRes = await fetch("http://localhost:8000/api/predict-exam", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          notes: uploadData.extracted_notes,
          topic_historical_frequencies: dummyHistories
        })
      });
      const predictData = await predictRes.json();
      if (!predictRes.ok) throw new Error(predictData.detail || "LLM Engine Crash");
      
      setChatHistory(prev => [...prev, {
        role: 'assistant',
        data: predictData.data,
        notesPayload: uploadData.extracted_notes
      }]);
    } catch (error) {
      console.error(error);
      setChatHistory(prev => [...prev, {
        role: 'assistant',
        content: `❌ Exception Failed: ${error.message}`
      }]);
    } finally {
      setLoading(false);
    }
  };

  const handleRegenerate = async (notesArray) => {
    setLoading(true);
    setChatHistory(prev => [...prev, {
      role: 'user',
      content: `🔄 Rerun pipeline logic on previous dataset vectors to dynamically synthesize mathematically alternative outputs.`
    }]);

    try {
      const dummyHistories = { "Backend Topic Flag Ignore": { 2024: 1 } };
      const predictRes = await fetch("http://localhost:8000/api/predict-exam", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          notes: notesArray,
          topic_historical_frequencies: dummyHistories
        })
      });
      const predictData = await predictRes.json();
      if (!predictRes.ok) throw new Error(predictData.detail || "LLM Engine Crash");
      
      setChatHistory(prev => [...prev, {
        role: 'assistant',
        data: predictData.data,
        notesPayload: notesArray
      }]);
    } catch (error) {
      setChatHistory(prev => [...prev, {
        role: 'assistant',
        content: `❌ Regeneration Failed: ${error.message}`
      }]);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="chat-app-container">
      <header className="chat-header">
        <h2>AI Exam Copilot UI</h2>
        <p className="status-indicator">● Model Pipeline Securely Connected</p>
      </header>

      <div className="chat-feed">
        {chatHistory.map((msg, index) => (
          <div key={index} className={`chat-message ${msg.role}`}>
            <div className={`message-avatar ${msg.role}`}>
              {msg.role === 'assistant' ? '🤖' : '👤'}
            </div>
            
            <div className="message-content" style={{ width: '100%' }}>
              {msg.content && <p>{msg.content}</p>}
              
              {msg.files && msg.files.length > 0 && (
                <div className="user-files-badge">
                  <strong>📎 Attached Volumes: </strong>
                  {msg.files.join(', ')}
                </div>
              )}

              {msg.data && <MLResultsBlock msg={msg} />}
            </div>
          </div>
        ))}
        
        {loading && (
          <div className="chat-message assistant">
            <div className="message-avatar assistant">🤖</div>
            <div className="message-content">
              <div className="typing-indicator">
                <span></span><span></span><span></span>
              </div>
              <p style={{ fontSize: '0.85rem', color: 'var(--text-muted)' }}>Executing multi-layer embedding vectors & unlocking deep length tokens natively...</p>
            </div>
          </div>
        )}
        
        {/* Floating Regenerate Button (ChatGPT Style) */}
        {!loading && chatHistory.length > 1 && chatHistory.slice(-1)[0].role === 'assistant' && chatHistory.slice(-1)[0].notesPayload && (
          <div className="regenerate-floating-container">
            <button 
              className="regenerate-floating-btn"
              onClick={() => handleRegenerate(chatHistory.slice(-1)[0].notesPayload)}
            >
              🔄 Regenerate response
            </button>
          </div>
        )}
        
        <div ref={chatEndRef} />
      </div>

      <div className="chat-input-area">
        {stagedFiles.length > 0 && (
          <div className="staged-files-tray">
            {stagedFiles.map((file, idx) => (
              <div key={idx} className="staged-file-chip">
                <span>{file.name} ({(file.size/1024/1024).toFixed(1)}MB)</span>
                <button title="Remove" onClick={() => removeStagedFile(idx)}>✕</button>
              </div>
            ))}
          </div>
        )}
        
        <div className="input-toolbar">
          <input 
            type="file" 
            ref={fileInputRef} 
            onChange={handleFileChange} 
            accept=".pdf" 
            multiple 
            style={{ display: 'none' }} 
          />
          <button className="upload-btn" onClick={() => fileInputRef.current.click()}>
            🔗 Upload PDFs
          </button>
          
          <div className="input-stats">
            Staged: <strong style={{color:"white"}}>{stagedFiles.length} / 10</strong> limits | 
            Max 50MB System Bound
          </div>
          
          <button 
            className="send-btn" 
            onClick={handleSend}
            disabled={stagedFiles.length === 0 || loading}
          >
            {loading ? 'Synthesizing...' : '🚀 Execute Analysis'}
          </button>
        </div>
      </div>
    </div>
  );
}

export default App;
