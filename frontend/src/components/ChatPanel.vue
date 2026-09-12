<script setup>
import { ref, nextTick, watch, computed } from 'vue'
import { useChatStore } from '../stores/chat'
import MessageBubble from './MessageBubble.vue'
import { DOC_TYPES, docColor } from '../utils/docTypes'
import { DEPARTMENTS, deptHint } from '../utils/departments'

const store = useChatStore()
const inputText = ref('')
const streamRef = ref(null)

// 参考来源去重（按文件名），用于答案下方展示引用
const uniqueSources = computed(() => {
  const seen = new Set()
  const out = []
  for (const s of store.sources) {
    const key = s.source || s.doc_type
    if (key && !seen.has(key)) {
      seen.add(key)
      out.push(s)
    }
  }
  return out
})

const EXAMPLES = [
  '公司年假有多少天？',
  '请假审批流程的第一步是什么？',
  '差旅费住宿标准是多少？',
  '密码长度最低要求是多少位？',
]

function scrollToBottom() {
  nextTick(() => {
    if (streamRef.value) streamRef.value.scrollTop = streamRef.value.scrollHeight
  })
}

watch(() => store.messages.length, scrollToBottom)
watch(() => store.messages[store.messages.length - 1]?.content, scrollToBottom)

function handleKeydown(e) {
  if (e.key === 'Enter' && !e.shiftKey) {
    e.preventDefault()
    handleSend()
  }
}

function handleSend() {
  const q = inputText.value.trim()
  if (!q) return
  store.send(q)
  inputText.value = ''
  nextTick(() => {
    const el = document.querySelector('.composer-input')
    if (el) el.style.height = 'auto'
  })
}

function useExample(q) {
  store.send(q)
}

function autoResize(e) {
  const el = e.target
  el.style.height = 'auto'
  el.style.height = Math.min(el.scrollHeight, 140) + 'px'
}
</script>

<template>
  <div class="chat">
    <div v-if="!store.messages.length" class="chat-empty">
      <div class="empty-card">
        <div class="empty-tabs">
          <span
            v-for="d in DOC_TYPES"
            :key="d.name"
            class="empty-tab"
            :style="{ background: d.color }"
          />
        </div>
        <h1 class="empty-title">企业知识库智能问答</h1>
        <p class="empty-sub">Enterprise Knowledge Base · Retrieval-Augmented Generation</p>
        <div class="empty-rule" />
        <p class="empty-label">试问 / SAMPLE QUERIES</p>
        <div class="examples">
          <button v-for="q in EXAMPLES" :key="q" class="example" @click="useExample(q)">
            {{ q }}
          </button>
        </div>
      </div>
    </div>

    <div v-else ref="streamRef" class="chat-stream">
      <MessageBubble
        v-for="(msg, i) in store.messages"
        :key="i"
        :role="msg.role"
        :content="msg.content"
        :streaming="store.streaming && i === store.messages.length - 1"
      />
      <div
        v-if="store.streaming && !store.messages[store.messages.length - 1]?.content"
        class="pending"
      >
        检索中
      </div>

      <div v-if="!store.streaming && uniqueSources.length" class="answer-sources">
        <span class="answer-sources-label">参考来源</span>
        <span
          v-for="(s, i) in uniqueSources"
          :key="i"
          class="answer-source-chip"
          :style="{ borderLeftColor: docColor(s.doc_type) }"
        >{{ s.source || s.doc_type }}</span>
      </div>
    </div>

    <div class="composer">
      <div class="composer-bar">
        <label class="dept">
          <span class="dept-label">当前部门</span>
          <select v-model="store.department" class="dept-select" :disabled="store.streaming">
            <option v-for="d in DEPARTMENTS" :key="d" :value="d">{{ d }}</option>
          </select>
        </label>
        <span class="dept-hint">{{ deptHint(store.department) }}</span>
      </div>
      <div class="composer-inner">
        <textarea
          v-model="inputText"
          class="composer-input"
          placeholder="输入问题，回车发送"
          rows="1"
          :disabled="store.streaming"
          @keydown="handleKeydown"
          @input="autoResize"
        />
        <button
          class="composer-send"
          :disabled="store.streaming || !inputText.trim()"
          @click="handleSend"
        >
          ➤
        </button>
      </div>
    </div>
  </div>
</template>
