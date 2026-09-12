<script setup>
import { ref } from 'vue'
import { useAuthStore } from '../stores/auth'
import { login, register } from '../api/index'
import { DOC_TYPES } from '../utils/docTypes'
import { DEPARTMENTS } from '../utils/departments'

const auth = useAuthStore()
const mode = ref('login')          // login | register
const username = ref('')
const password = ref('')
const department = ref('公共')
const loading = ref(false)
const error = ref('')

function switchMode(m) {
  mode.value = m
  error.value = ''
}

async function submit() {
  if (!username.value || !password.value) {
    error.value = '请输入用户名和密码'
    return
  }
  loading.value = true
  error.value = ''
  try {
    const data = mode.value === 'login'
      ? await login(username.value, password.value)
      : await register(username.value, password.value, department.value)
    auth.setAuth(data)
  } catch (e) {
    error.value = e.response?.data?.detail
      || (mode.value === 'login' ? '登录失败，请重试' : '注册失败，请重试')
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
      <p class="login-sub">Enterprise Knowledge Base · {{ mode === 'login' ? 'Sign in' : 'Sign up' }}</p>
      <div class="login-rule" />

      <div class="login-switch">
        <button :class="{ active: mode === 'login' }" @click="switchMode('login')">登录</button>
        <button :class="{ active: mode === 'register' }" @click="switchMode('register')">注册</button>
      </div>

      <label class="login-field">
        <span class="login-label">用户名</span>
        <input v-model="username" type="text" placeholder="3-32 位" @keydown.enter="submit" />
      </label>

      <label class="login-field">
        <span class="login-label">密码</span>
        <input v-model="password" type="password" placeholder="至少 6 位" @keydown.enter="submit" />
      </label>

      <label v-if="mode === 'register'" class="login-field">
        <span class="login-label">部门</span>
        <select v-model="department" class="login-select">
          <option v-for="d in DEPARTMENTS" :key="d" :value="d">{{ d }}</option>
        </select>
      </label>

      <p v-if="error" class="login-error">{{ error }}</p>

      <button class="login-btn" :disabled="loading" @click="submit">
        {{ loading ? '处理中' : (mode === 'login' ? '登 录' : '注 册') }}
      </button>
      <p class="login-hint">
        {{ mode === 'login' ? '默认管理员：admin / admin123' : '注册后默认仅可见「公共」及所选部门文档' }}
      </p>
    </div>
  </div>
</template>
