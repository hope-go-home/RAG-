import { defineStore } from 'pinia'
import { ref, computed } from 'vue'

const TOKEN_KEY = 'kb_token'
const USER_KEY = 'kb_user'

function readStorage(key) {
  try {
    return localStorage.getItem(key)
  } catch {
    return null
  }
}

function writeStorage(key, value) {
  try {
    localStorage.setItem(key, value)
  } catch {
    // 隐私模式 / 存储被禁用时静默失败
  }
}

function removeStorage(key) {
  try {
    localStorage.removeItem(key)
  } catch {
    // ignore
  }
}

export const useAuthStore = defineStore('auth', () => {
  const token = ref(readStorage(TOKEN_KEY) || '')
  const user = ref(JSON.parse(readStorage(USER_KEY) || 'null'))

  const isAdmin = computed(() => user.value?.role === 'admin')
  const department = computed(() => user.value?.department || '公共')

  function setAuth(data) {
    token.value = data.token
    user.value = {
      username: data.username,
      department: data.department,
      role: data.role,
    }
    writeStorage(TOKEN_KEY, token.value)
    writeStorage(USER_KEY, JSON.stringify(user.value))
  }

  function setUser(u) {
    user.value = {
      username: u.username,
      department: u.department,
      role: u.role,
    }
    writeStorage(USER_KEY, JSON.stringify(user.value))
  }

  function logout() {
    token.value = ''
    user.value = null
    removeStorage(TOKEN_KEY)
    removeStorage(USER_KEY)
  }

  return { token, user, isAdmin, department, setAuth, setUser, logout }
})
