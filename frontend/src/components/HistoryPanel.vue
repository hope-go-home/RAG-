<script setup>
import { ref, onMounted, watch } from 'vue'
import { useChatStore } from '../stores/chat'
import { getSessions } from '../api/index'

const store = useChatStore()
const sessions = ref([])
const open = ref(true)

async function load() {
  try {
    const data = await getSessions()
    sessions.value = Array.isArray(data) ? data : []
  } catch {
    sessions.value = []
  }
}

onMounted(load)
watch(() => store.streaming, (val) => { if (!val) load() })

function formatTime(ts) {
  if (!ts) return ''
  const d = new Date(String(ts).replace(' ', 'T'))
  if (Number.isNaN(d.getTime())) return ts
  return d.toLocaleString('zh-CN', {
    month: '2-digit',
    day: '2-digit',
    hour: '2-digit',
    minute: '2-digit',
  })
}
</script>

<template>
  <section class="panel">
    <header class="panel-head">
      <h3 class="panel-label">历史会话</h3>
      <button class="panel-meta" @click="open = !open">{{ open ? '收起' : '展开' }}</button>
    </header>
    <div v-if="open" class="panel-body" style="padding: 4px 10px 8px">
      <div v-if="sessions.length" class="hist">
        <button
          v-for="s in sessions"
          :key="s.session_id"
          :class="['hist-row', { active: s.session_id === store.sessionId }]"
          @click="store.loadSession(s.session_id)"
        >
          <span class="hist-q">{{ s.first_question }}</span>
          <span class="hist-time">{{ formatTime(s.last_time) }} · {{ s.count }} 轮</span>
        </button>
      </div>
      <div v-else class="note">暂无历史会话</div>
    </div>
  </section>
</template>
