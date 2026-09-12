<script setup>
import HomeView from './views/HomeView.vue'
import LoginView from './views/LoginView.vue'
import { useAuthStore } from './stores/auth'

const auth = useAuthStore()
</script>

<template>
  <LoginView v-if="!auth.token" />
  <HomeView v-else />
</template>
