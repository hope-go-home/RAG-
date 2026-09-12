<script setup>
import { onMounted } from 'vue'
import HomeView from './views/HomeView.vue'
import LoginView from './views/LoginView.vue'
import { useAuthStore } from './stores/auth'
import { getMe } from './api/index'

const auth = useAuthStore()

// 启动校验：本地有 token 时向服务端确认有效性；无效则清除（401 由拦截器处理）
onMounted(async () => {
  if (!auth.token) return
  try {
    const me = await getMe()
    auth.setUser(me)
  } catch {
    // 网络异常时保留登录态，避免误登出
  }
})
</script>

<template>
  <LoginView v-if="!auth.token" />
  <HomeView v-else />
</template>
