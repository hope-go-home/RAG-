<script setup>
import { ref, nextTick, watch, computed } from 'vue'
import { useChatStore } from '../stores/chat'
import MessageBubble from './MessageBubble.vue'
import SourceCard from './SourceCard.vue'
import Feedback from './Feedback.vue'
import { t } from '../i18n'

const store = useChatStore()
const inputText = ref('')
const streamRef = ref(null)

const examples = computed(() => t('chat.examples'))

const lastUserQuestion = computed(() => {
  for (let i = store.messages.length - 1; i >= 0; i--) {
    if (store.messages[i].role === 'user') return store.messages[i].content
  }
  return ''
})
const lastAnswer = computed(() => {
  for (let i = store.messages.length - 1; i >= 0; i--) {
    if (store.messages[i].role === 'assistant') return store.messages[i].content
  }
  return ''
})

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
  el.style.height = Math.min(el.scrollHeight, 160) + 'px'
}
</script>

<template>
  <div class="chat">
    <div v-if="!store.messages.length" class="chat-empty">
      <div class="empty-card">
        <div class="empty-eyebrow">WIX HELP CENTER · HYBRID RAG</div>
        <h1 class="empty-title">{{ t('chat.heroTitle') }}</h1>
        <p class="empty-sub">{{ t('chat.heroSub') }}</p>
        <div class="empty-rule" />
        <p class="empty-label">{{ t('chat.examplesLabel') }}</p>
        <div class="examples">
          <button v-for="q in examples" :key="q" class="example" @click="useExample(q)">
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

      <div v-if="!store.streaming && store.sources.length" class="answer-sources">
        <div class="answer-sources-head">{{ t('chat.reference') }}</div>
        <div class="answer-sources-list">
          <SourceCard
            v-for="(s, i) in store.sources"
            :key="i"
            :item="s"
            :index="i"
          />
        </div>
        <Feedback
          :question="lastUserQuestion"
          :answer="lastAnswer"
          :session-id="store.sessionId"
        />
      </div>
    </div>

    <div class="composer">
      <div class="composer-inner">
        <textarea
          v-model="inputText"
          class="composer-input"
          :placeholder="t('chat.placeholder')"
          rows="1"
          :disabled="store.streaming"
          @keydown="handleKeydown"
          @input="autoResize"
        />
        <button
          class="composer-send"
          :disabled="store.streaming || !inputText.trim()"
          @click="handleSend"
        >{{ t('chat.send') }}</button>
      </div>
    </div>
  </div>
</template>
