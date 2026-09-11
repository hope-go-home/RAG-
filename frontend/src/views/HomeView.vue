<script setup>
import { ref } from 'vue'
import ThinkPanel from '../components/ThinkPanel.vue'
import StatsPanel from '../components/StatsPanel.vue'
import TracePanel from '../components/TracePanel.vue'
import UploadPanel from '../components/UploadPanel.vue'
import HistoryPanel from '../components/HistoryPanel.vue'
import ChatPanel from '../components/ChatPanel.vue'
import { useChatStore } from '../stores/chat'
import { DOC_TYPES } from '../utils/docTypes'

const store = useChatStore()

const MIN_W = 280
const MAX_W = 720
const WIDTH_KEY = 'kb_sidebar_width'

function clamp(w) {
  return Math.min(Math.max(w, MIN_W), MAX_W)
}

const sidebarWidth = ref(clamp(Number(localStorage.getItem(WIDTH_KEY)) || 380))

function startResize(e) {
  e.preventDefault()
  const startX = e.clientX
  const startW = sidebarWidth.value

  function onMove(ev) {
    sidebarWidth.value = clamp(startW + ev.clientX - startX)
  }
  function onUp() {
    document.removeEventListener('mousemove', onMove)
    document.removeEventListener('mouseup', onUp)
    document.body.style.userSelect = ''
    document.body.style.cursor = ''
    localStorage.setItem(WIDTH_KEY, String(sidebarWidth.value))
  }

  document.body.style.userSelect = 'none'
  document.body.style.cursor = 'col-resize'
  document.addEventListener('mousemove', onMove)
  document.addEventListener('mouseup', onUp)
}
</script>

<template>
  <div class="app-layout">
    <aside class="sidebar" :style="{ '--sidebar-w': sidebarWidth + 'px' }">
      <header class="masthead">
        <div class="masthead-top">
          <div>
            <div class="masthead-title">知识库</div>
            <div class="masthead-sub">Enterprise Knowledge Base</div>
          </div>
          <button class="new-session" title="清空当前对话，开启新会话" @click="store.reset()">
            ＋ 新会话
          </button>
        </div>
        <div class="legend">
          <span v-for="d in DOC_TYPES" :key="d.name" class="legend-chip">
            <span class="legend-dot" :style="{ background: d.color }" />
            {{ d.short }}
          </span>
        </div>
      </header>
      <div class="sidebar-scroll">
        <ThinkPanel />
        <StatsPanel />
        <TracePanel />
        <UploadPanel />
        <HistoryPanel />
      </div>
    </aside>

    <div class="resizer" title="拖动调节宽度" @mousedown="startResize" />

    <main class="main-area">
      <ChatPanel />
    </main>
  </div>
</template>
