<script setup>
import { ref } from 'vue'
import { useAuthStore } from '../stores/auth'
import { login, register } from '../api'
import { t } from '../i18n'
import LanguageToggle from '../components/LanguageToggle.vue'

const auth = useAuthStore()
const mode = ref('login') // login | register
const username = ref('')
const password = ref('')
const loading = ref(false)
const error = ref('')

function switchMode(m) {
  mode.value = m
  error.value = ''
}

async function submit() {
  if (!username.value || !password.value) {
    error.value = t('login.errEmpty')
    return
  }
  loading.value = true
  error.value = ''
  try {
    const data = mode.value === 'login'
      ? await login(username.value, password.value)
      : await register(username.value, password.value)
    auth.setAuth(data)
  } catch (e) {
    error.value = e.response?.data?.detail
      || (mode.value === 'login' ? t('login.errLogin') : t('login.errRegister'))
  } finally {
    loading.value = false
  }
}
</script>

<template>
  <div class="login-wrap">
    <div class="login-lang"><LanguageToggle /></div>
    <div class="login-shell">
      <aside class="login-aside">
        <div>
          <span class="brand-mark" aria-hidden="true">
            <svg viewBox="0 0 32 32" width="26" height="26"><path d="M7 10l4 13 3-9 3 9 4-13" fill="none" stroke="currentColor" stroke-width="2.6" stroke-linecap="round" stroke-linejoin="round"/></svg>
          </span>
          <h2>{{ t('app.title') }}</h2>
          <p>{{ t('app.subtitle') }}</p>
          <div class="login-chain">
            <div class="row">
              <span class="chip d">D#3</span>
              <span class="chip s">S#7</span>
              <span>→ RRF →</span>
              <span class="chip r">↑ 0.98</span>
            </div>
            <div class="row">dense · sparse · fusion · rerank</div>
          </div>
        </div>
        <ul class="login-points">
          <li>{{ t('login.point1') }}</li>
          <li>{{ t('login.point2') }}</li>
          <li>{{ t('login.point3') }}</li>
        </ul>
      </aside>

      <div class="login-form">
        <h1 class="login-title">{{ mode === 'login' ? t('login.signin') : t('login.signup') }}</h1>
        <p class="login-sub">{{ t('login.sub') }}</p>
        <div class="login-rule" />

        <div class="login-switch">
          <button :class="{ active: mode === 'login' }" @click="switchMode('login')">{{ t('login.signin') }}</button>
          <button :class="{ active: mode === 'register' }" @click="switchMode('register')">{{ t('login.signup') }}</button>
        </div>

        <label class="login-field">
          <span class="login-label">{{ t('login.username') }}</span>
          <input v-model="username" type="text" :placeholder="t('login.usernamePh')" @keydown.enter="submit" />
        </label>

        <label class="login-field">
          <span class="login-label">{{ t('login.password') }}</span>
          <input v-model="password" type="password" :placeholder="t('login.passwordPh')" @keydown.enter="submit" />
        </label>

        <p v-if="error" class="login-error">{{ error }}</p>

        <button class="login-btn" :disabled="loading" @click="submit">
          {{ loading ? '…' : (mode === 'login' ? t('login.submitLogin') : t('login.submitRegister')) }}
        </button>
        <p class="login-hint">{{ t('login.hint') }}</p>
      </div>
    </div>
  </div>
</template>
