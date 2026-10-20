import React, { useEffect, useState } from 'react'

export default function App() {
  const [status, setStatus] = useState<string>('Checking backend...')

  useEffect(() => {
    fetch('http://localhost:8000/health')
      .then(res => res.json())
      .then(data => setStatus(`Backend status: ${data.status} (${data.service})`))
      .catch(_ => setStatus('Backend offline'))
  }, [])

  return (
    <div style={{ fontFamily: 'sans-serif', textAlign: 'center', marginTop: '50px' }}>
      <h1>CrewDesk Research Platform</h1>
      <p>{status}</p>
    </div>
  )
}
