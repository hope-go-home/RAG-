<script setup>
import { ref } from 'vue'
import ThinkPanel from '../components/ThinkPanel.vue'
import StatsPanel from '../components/StatsPanel.vue'
import TracePanel from '../components/TracePanel.vue'
import TypeFilter from '../components/TypeFilter.vue'
import UploadPanel from '../components/UploadPanel.vue'
import DocumentPanel from '../components/DocumentPanel.vue'
import UserPanel from '../components/UserPanel.vue'
import HistoryPanel from '../components/HistoryPanel.vue'
import ChatPanel from '../components/ChatPanel.vue'
import EvaluationView from '../components/EvaluationView.vue'
import LanguageToggle from '../components/LanguageToggle.vue'
import { useChatStore } from '../stores/chat'
import { useAuthStore } from '../stores/auth'
import { t } from '../i18n'

const store = useChatStore()
const auth = useAuthStore()
const showEval = ref(false)

function handleLogout() {
  store.reset()
  auth.logout()
}

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
          <div class="masthead-brand">
            <span class="brand-mark" aria-hidden="true">
              <svg viewBox="0 0 32 32" width="26" height="26"><path d="M7 10l4 13 3-9 3 9 4-13" fill="none" stroke="currentColor" stroke-width="2.6" stroke-linecap="round" stroke-linejoin="round"/></svg>
            </span>
            <div>
              <div class="masthead-title">{{ t('app.title') }}</div>
              <div class="masthead-sub">{{ auth.user?.username }} · {{ auth.user?.role }}</div>
            </div>
          </div>
          <div class="masthead-actions">
            <button class="icon-btn" :title="t('header.newSession')" @click="store.reset()">＋</button>
            <button class="icon-btn" :title="t('header.logout')" @click="handleLogout">⎋</button>
          </div>
        </div>
        <div class="masthead-bar">
          <LanguageToggle />
          <button class="link-btn" @click="showEval = true">{{ t('evaluation.open') }}</button>
        </div>
      </header>
      <div class="sidebar-scroll">
        <ThinkPanel />
        <StatsPanel />
        <TracePanel />
        <TypeFilter />
        <UploadPanel v-if="auth.isAdmin" />
        <DocumentPanel v-if="auth.isAdmin" />
        <UserPanel v-if="auth.isAdmin" />
        <HistoryPanel />
      </div>
    </aside>

    <div class="resizer" :title="t('header.language')" @mousedown="startResize" />

    <main class="main-area">
      <ChatPanel />
    </main>

    <EvaluationView :open="showEval" @close="showEval = false" />
  </div>
</template>
