export async function streamChat({ question, sessionId, department }, onEvent) {
  const response = await fetch('/chat/stream', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ question, session_id: sessionId, department }),
  })

  if (!response.ok) {
    const text = await response.text()
    onEvent({ type: 'error', data: { message: text || '请求失败' } })
    return
  }

  const reader = response.body.getReader()
  const decoder = new TextDecoder()
  let buffer = ''

  while (true) {
    const { done, value } = await reader.read()
    if (done) break

    buffer += decoder.decode(value, { stream: true })
    const lines = buffer.split('\n')
    buffer = lines.pop()

    for (const line of lines) {
      const trimmed = line.trim()
      if (!trimmed || !trimmed.startsWith('data:')) continue
      const jsonStr = trimmed.slice(5).trim()
      if (!jsonStr || jsonStr === '[DONE]') continue
      try {
        const event = JSON.parse(jsonStr)
        onEvent(event)
      } catch {
        // skip malformed lines
      }
    }
  }

  // Process remaining buffer
  if (buffer.trim()) {
    const trimmed = buffer.trim()
    if (trimmed.startsWith('data:')) {
      const jsonStr = trimmed.slice(5).trim()
      if (jsonStr && jsonStr !== '[DONE]') {
        try {
          const event = JSON.parse(jsonStr)
          onEvent(event)
        } catch {
          // skip
        }
      }
    }
  }
}
