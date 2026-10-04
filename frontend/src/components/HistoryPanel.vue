<script setup>
import { ref, onMounted, watch } from 'vue'
import { useChatStore } from '../stores/chat'
import { getSessions, deleteSession } from '../api'
import { t } from '../i18n'

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
  return d.toLocaleString(undefined, {
    month: '2-digit',
    day: '2-digit',
    hour: '2-digit',
    minute: '2-digit',
  })
}

async function handleDelete(s) {
  if (!confirm(t('history.confirm').replace('{q}', s.first_question))) return
  try {
    await deleteSession(s.session_id)
    if (s.session_id === store.sessionId) store.reset()
    await load()
  } catch {
    // 失败时忽略，保持列表
  }
}
</script>

<template>
  <section class="panel">
    <header class="panel-head">
      <h3 class="panel-label">{{ t('history.title') }}</h3>
      <button class="panel-meta" @click="open = !open">
        {{ open ? t('history.collapse') : t('history.expand') }}
      </button>
    </header>
    <div v-if="open" class="panel-body" style="padding: 4px 10px 10px">
      <div v-if="sessions.length" class="hist">
        <div
          v-for="s in sessions"
          :key="s.session_id"
          :class="['hist-row', { active: s.session_id === store.sessionId }]"
        >
          <button class="hist-main" @click="store.loadSession(s.session_id)">
            <span class="hist-q">{{ s.first_question }}</span>
            <span class="hist-time">{{ formatTime(s.last_time) }} · {{ s.count }} {{ t('history.turns') }}</span>
          </button>
          <button class="hist-del" :title="t('history.delete')" @click.stop="handleDelete(s)">×</button>
        </div>
      </div>
      <div v-else class="note">{{ t('history.empty') }}</div>
    </div>
  </section>
</template>
