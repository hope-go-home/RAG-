import { defineStore } from 'pinia'
import { ref } from 'vue'
import { streamChat } from '../api/stream'
import { getHistory } from '../api/index'
import { DEFAULT_DEPARTMENT } from '../utils/departments'

const STORAGE_KEY = 'kb_chat_state'

function genId() {
  return 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, (c) => {
    const r = (Math.random() * 16) | 0
    return (c === 'x' ? r : (r & 0x3) | 0x8).toString(16)
  })
}

function loadState() {
  try {
    const raw = localStorage.getItem(STORAGE_KEY)
    return raw ? JSON.parse(raw) : null
  } catch {
    return null
  }
}

export const useChatStore = defineStore('chat', () => {
  const saved = loadState()
  const sessionId = ref(saved?.sessionId || genId())
  const messages = ref(saved?.messages || [])
  const thinking = ref(saved?.thinking || [])
  const stats = ref(saved?.stats || null)
  const sources = ref(saved?.sources || [])
  const department = ref(saved?.department || DEFAULT_DEPARTMENT)
  const streaming = ref(false)
  const error = ref(null)

  function persist() {
    try {
      localStorage.setItem(STORAGE_KEY, JSON.stringify({
        sessionId: sessionId.value,
        messages: messages.value,
        thinking: thinking.value,
        stats: stats.value,
        sources: sources.value,
        department: department.value,
      }))
    } catch {
      // 存储超限时静默失败，不影响对话
    }
  }

  function reset() {
    sessionId.value = genId()
    messages.value = []
    thinking.value = []
    stats.value = null
    sources.value = []
    streaming.value = false
    error.value = null
    try {
      localStorage.removeItem(STORAGE_KEY)
    } catch {
      // ignore
    }
  }

  async function loadSession(sid) {
    if (streaming.value || sid === sessionId.value) return
    try {
      const records = await getHistory(sid)
      sessionId.value = sid
      messages.value = []
      thinking.value = []
      stats.value = null
      sources.value = []
      error.value = null
      for (const r of records) {
        messages.value.push({ role: 'user', content: r.question })
        messages.value.push({ role: 'assistant', content: r.answer })
      }
      persist()
    } catch (e) {
      error.value = e.message || '加载历史会话失败'
    }
  }

  function send(question) {
    if (!question.trim() || streaming.value) return

    messages.value.push({ role: 'user', content: question })
    messages.value.push({ role: 'assistant', content: '' })
    streaming.value = true
    thinking.value = []
    stats.value = null
    sources.value = []
    error.value = null

    const currentIdx = messages.value.length - 1

    streamChat({ question, sessionId: sessionId.value, department: department.value }, (event) => {
      switch (event.type) {
        case 'thinking':
          thinking.value.push({ node: event.node, info: event.info })
          break
        case 'stats':
          stats.value = event
          break
        case 'sources':
          sources.value = event.sources || []
          break
        case 'token':
          messages.value[currentIdx].content += event.content || ''
          break
        case 'answer':
          messages.value[currentIdx].content = event.answer || ''
          break
        case 'done':
          streaming.value = false
          persist()
          break
        case 'error':
          streaming.value = false
          error.value = event.message || '未知错误'
          messages.value[currentIdx].content = `⚠️ ${error.value}`
          persist()
          break
      }
    }).catch((e) => {
      streaming.value = false
      error.value = e.message || '网络错误'
      messages.value[currentIdx].content = `⚠️ ${error.value}`
      persist()
    })
  }

  return {
    sessionId,
    messages,
    thinking,
    stats,
    sources,
    department,
    streaming,
    error,
    send,
    reset,
    loadSession,
  }
})
