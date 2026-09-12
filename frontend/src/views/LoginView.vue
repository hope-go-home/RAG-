<script setup>
import { ref } from 'vue'
import { useAuthStore } from '../stores/auth'
import { login } from '../api/index'
import { DOC_TYPES } from '../utils/docTypes'

const auth = useAuthStore()
const username = ref('')
const password = ref('')
const loading = ref(false)
const error = ref('')

async function handleLogin() {
  if (!username.value || !password.value) {
    error.value = '请输入用户名和密码'
    return
  }
  loading.value = true
  error.value = ''
  try {
    const data = await login(username.value, password.value)
    auth.setAuth(data)
  } catch (e) {
    error.value = e.response?.data?.detail || '登录失败，请重试'
  } finally {
    loading.value = false
  }
}
</script>

<template>
  <div class="login-wrap">
    <div class="login-card">
      <div class="login-tabs">
        <span
          v-for="d in DOC_TYPES"
          :key="d.name"
          class="login-tab"
          :style="{ background: d.color }"
        />
      </div>
      <h1 class="login-title">企业知识库智能问答</h1>
      <p class="login-sub">Enterprise Knowledge Base · Sign in</p>
      <div class="login-rule" />

      <label class="login-field">
        <span class="login-label">用户名</span>
        <input
          v-model="username"
          type="text"
          placeholder="username"
          autocomplete="username"
          @keydown.enter="handleLogin"
        />
      </label>

      <label class="login-field">
        <span class="login-label">密码</span>
        <input
          v-model="password"
          type="password"
          placeholder="password"
          autocomplete="current-password"
          @keydown.enter="handleLogin"
        />
      </label>

      <p v-if="error" class="login-error">{{ error }}</p>

      <button class="login-btn" :disabled="loading" @click="handleLogin">
        {{ loading ? '登录中' : '登 录' }}
      </button>
      <p class="login-hint">默认管理员：admin / admin123</p>
    </div>
  </div>
</template>
