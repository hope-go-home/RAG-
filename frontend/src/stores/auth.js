import { defineStore } from 'pinia'
import { ref, computed } from 'vue'

const TOKEN_KEY = 'kb_token'
const USER_KEY = 'kb_user'

export const useAuthStore = defineStore('auth', () => {
  const token = ref(localStorage.getItem(TOKEN_KEY) || '')
  const user = ref(JSON.parse(localStorage.getItem(USER_KEY) || 'null'))

  const isAdmin = computed(() => user.value?.role === 'admin')
  const department = computed(() => user.value?.department || '公共')

  function setAuth(data) {
    token.value = data.token
    user.value = {
      username: data.username,
      department: data.department,
      role: data.role,
    }
    localStorage.setItem(TOKEN_KEY, token.value)
    localStorage.setItem(USER_KEY, JSON.stringify(user.value))
  }

  function logout() {
    token.value = ''
    user.value = null
    localStorage.removeItem(TOKEN_KEY)
    localStorage.removeItem(USER_KEY)
  }

  return { token, user, isAdmin, department, setAuth, logout }
})
