import React, { useState, useEffect, useRef } from 'react'

interface EventItem {
  event_id: string
  event_type: string
  timestamp: string
  data: any
}

export default function App() {
  const [topic, setTopic] = useState('')
  const [runId, setRunId] = useState<string | null>(null)
  const [runStatus, setRunStatus] = useState<string>('idle')
  const [events, setEvents] = useState<EventItem[]>([])
  const [nodeStatuses, setNodeStatuses] = useState<Record<string, string>>({
    Planner: 'pending',
    Researcher: 'pending',
    Writer: 'pending',
    Critic: 'pending',
    Finalizer: 'pending'
  })
  const [revisionCount, setRevisionCount] = useState<number>(0)
  const [llmUsage, setLlmUsage] = useState<number>(0)
  const [maxLlm, setMaxLlm] = useState<number>(25)
  const [report, setReport] = useState<any>(null)
  const [summary, setSummary] = useState<any>(null)
  const [errorMsg, setErrorMsg] = useState<string | null>(null)
  const eventSourceRef = useRef<EventSource | null>(null)

  const startResearch = async (e: React.FormEvent) => {
    e.preventDefault()
    if (!topic.trim()) return

    setErrorMsg(null)
    setEvents([])
    setReport(null)
    setSummary(null)
    setRevisionCount(0)
    setLlmUsage(0)
    setNodeStatuses({
      Planner: 'running',
      Researcher: 'pending',
      Writer: 'pending',
      Critic: 'pending',
      Finalizer: 'pending'
    })

    try {
      const res = await fetch('http://localhost:8000/api/runs', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ topic })
      })
      if (!res.ok) throw new Error('Failed to start run')
      const data = await res.json()
      setRunId(data.run_id)
      setRunStatus('running')
      connectSSE(data.run_id)
    } catch (err: any) {
      setErrorMsg(err.message || 'Error starting run')
      setRunStatus('failed')
    }
  }

  const connectSSE = (id: string) => {
    if (eventSourceRef.current) {
      eventSourceRef.current.close()
    }

    const es = new EventSource(`http://localhost:8000/api/runs/${id}/events`)
    eventSourceRef.current = es

    const eventTypes = [
      'run_started',
      'agent_started',
      'agent_progress',
      'agent_completed',
      'graph_transition',
      'critic_decision',
      'budget_warning',
      'run_completed',
      'run_failed'
    ]

    eventTypes.forEach(type => {
      es.addEventListener(type, (e: MessageEvent) => {
        try {
          const parsed = JSON.parse(e.data)
          setEvents(prev => [parsed, ...prev])

          if (parsed.event_type === 'agent_started') {
            const agent = parsed.data.agent_name?.replace('Agent', '')
            if (agent && nodeStatuses[agent] !== undefined) {
              setNodeStatuses(prev => ({ ...prev, [agent]: 'running' }))
            }
          } else if (parsed.event_type === 'agent_completed') {
            const agent = parsed.data.agent_name?.replace('Agent', '')
            if (agent) {
              setNodeStatuses(prev => ({ ...prev, [agent]: 'completed' }))
            }
          } else if (parsed.event_type === 'graph_transition') {
            const dest = parsed.data.destination
            if (dest === 'Planner') setNodeStatuses({ Planner: 'completed', Researcher: 'running', Writer: 'pending', Critic: 'pending', Finalizer: 'pending' })
            if (dest === 'Researcher') setNodeStatuses({ Planner: 'completed', Researcher: 'completed', Writer: 'running', Critic: 'pending', Finalizer: 'pending' })
            if (dest === 'Writer') setNodeStatuses({ Planner: 'completed', Researcher: 'completed', Writer: 'completed', Critic: 'running', Finalizer: 'pending' })
            if (dest === 'Critic') setNodeStatuses({ Planner: 'completed', Researcher: 'completed', Writer: 'completed', Critic: 'completed', Finalizer: 'pending' })
            if (dest === 'Finalizer') setNodeStatuses({ Planner: 'completed', Researcher: 'completed', Writer: 'completed', Critic: 'completed', Finalizer: 'completed' })
            if (parsed.data.revision_count !== undefined) {
              setRevisionCount(parsed.data.revision_count)
            }
          } else if (parsed.event_type === 'run_completed') {
            setRunStatus('completed')
            setSummary(parsed.data.summary)
            fetchReport(id)
            es.close()
          } else if (parsed.event_type === 'run_failed' || parsed.event_type === 'budget_warning') {
            setRunStatus('failed')
            setErrorMsg(parsed.data.error || 'Run failed or budget exhausted')
            es.close()
          }
        } catch (err) {
          console.error('Error parsing SSE event', err)
        }
      })
    })

    es.onerror = () => {
      console.warn('SSE connection lost. Reconnecting...')
    }
  }

  const fetchReport = async (id: string) => {
    try {
      const res = await fetch(`http://localhost:8000/api/runs/${id}/report`)
      if (res.ok) {
        const data = await res.json()
        setReport(data)
      }
    } catch (err) {
      console.error('Failed to fetch report', err)
    }
  }

  useEffect(() => {
    return () => {
      if (eventSourceRef.current) {
        eventSourceRef.current.close()
      }
    }
  }, [])

  return (
    <div style={{ fontFamily: 'system-ui, sans-serif', maxWidth: '1200px', margin: '0 auto', padding: '20px', color: '#333' }}>
      <header style={{ borderBottom: '1px solid #eaeaea', paddingBottom: '20px', marginBottom: '20px' }}>
        <h1 style={{ margin: '0 0 10px 0', color: '#1a365d' }}>CrewDesk Research Platform</h1>
        <p style={{ margin: 0, color: '#666' }}>Production-minded multi-agent research report generation</p>
      </header>

      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '20px' }}>
        <div style={{ background: '#f8fafc', padding: '20px', borderRadius: '8px', border: '1px solid #e2e8f0' }}>
          <h2>Start New Research Run</h2>
          <form onSubmit={startResearch}>
            <div style={{ marginBottom: '15px' }}>
              <label style={{ display: 'block', marginBottom: '5px', fontWeight: 600 }}>Research Topic:</label>
              <input
                type="text"
                value={topic}
                onChange={e => setTopic(e.target.value)}
                placeholder="e.g. Impact of Generative AI on Software Engineering"
                style={{ width: '100%', padding: '10px', borderRadius: '4px', border: '1px solid #cbd5e1' }}
              />
            </div>
            <button
              type="submit"
              disabled={runStatus === 'running'}
              style={{
                background: runStatus === 'running' ? '#94a3b8' : '#2563eb',
                color: '#fff',
                border: 'none',
                padding: '10px 20px',
                borderRadius: '4px',
                fontWeight: 600,
                cursor: runStatus === 'running' ? 'not-allowed' : 'pointer'
              }}
            >
              {runStatus === 'running' ? 'Researching...' : 'Start Research'}
            </button>
          </form>

          {runId && (
            <div style={{ marginTop: '20px', fontSize: '14px' }}>
              <strong>Run ID:</strong> {runId} <br />
              <strong>Status:</strong> <span style={{ textTransform: 'uppercase', fontWeight: 600 }}>{runStatus}</span> <br />
              <strong>Revision Count:</strong> {revisionCount}/2 <br />
              <strong>LLM Budget Used:</strong> {llmUsage}/{maxLlm}
            </div>
          )}

          {errorMsg && (
            <div style={{ marginTop: '15px', padding: '10px', background: '#fee2e2', color: '#991b1b', borderRadius: '4px' }}>
              {errorMsg}
            </div>
          )}
        </div>

        <div style={{ background: '#f8fafc', padding: '20px', borderRadius: '8px', border: '1px solid #e2e8f0' }}>
          <h2>Workflow Graph Status</h2>
          <div style={{ display: 'flex', flexDirection: 'column', gap: '10px', marginTop: '15px' }}>
            {['Planner', 'Researcher', 'Writer', 'Critic', 'Finalizer'].map(node => {
              const status = nodeStatuses[node] || 'pending'
              const bg = status === 'completed' ? '#dcfce7' : status === 'running' ? '#dbeafe' : '#f1f5f9'
              const color = status === 'completed' ? '#166534' : status === 'running' ? '#1e40af' : '#64748b'
              return (
                <div key={node} style={{ display: 'flex', justifyContent: 'space-between', padding: '10px 15px', background: bg, color: color, borderRadius: '6px', fontWeight: 600, border: '1px solid #cbd5e1' }}>
                  <span>{node} Node</span>
                  <span style={{ textTransform: 'uppercase', fontSize: '12px' }}>{status}</span>
                </div>
              )
            })}
          </div>
        </div>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '20px', marginTop: '20px' }}>
        <div style={{ background: '#f8fafc', padding: '20px', borderRadius: '8px', border: '1px solid #e2e8f0', maxHeight: '400px', overflowY: 'auto' }}>
          <h2>Live Event Feed</h2>
          {events.length === 0 ? (
            <p style={{ color: '#64748b' }}>No events recorded yet.</p>
          ) : (
            events.map(ev => (
              <div key={ev.event_id} style={{ fontSize: '13px', padding: '8px', borderBottom: '1px solid #e2e8f0', marginBottom: '5px' }}>
                <span style={{ color: '#2563eb', fontWeight: 600 }}>[{ev.event_type}]</span> {new Date(ev.timestamp).toLocaleTimeString()}
                <div style={{ color: '#475569', fontSize: '12px', wordBreak: 'break-all' }}>{JSON.stringify(ev.data)}</div>
              </div>
            ))
          )}
        </div>

        <div style={{ background: '#f8fafc', padding: '20px', borderRadius: '8px', border: '1px solid #e2e8f0', maxHeight: '400px', overflowY: 'auto' }}>
          <h2>Completed Report & Summary</h2>
          {report ? (
            <div>
              <div style={{ whiteSpace: 'pre-wrap', background: '#fff', padding: '15px', borderRadius: '6px', border: '1px solid #e2e8f0', fontSize: '14px' }}>
                {report.markdown_content}
              </div>
              {summary && (
                <div style={{ marginTop: '15px', fontSize: '13px', color: '#475569' }}>
                  <strong>Duration:</strong> {summary.total_duration_seconds?.toFixed(2)}s | 
                  <strong> Sources:</strong> {summary.source_count} | 
                  <strong> LLM Calls:</strong> {summary.llm_call_count}
                </div>
              )}
            </div>
          ) : (
            <p style={{ color: '#64748b' }}>Report will appear here once research run completes.</p>
          )}
        </div>
      </div>
    </div>
  )
}
