<script setup>
import { ref, onMounted } from 'vue'
import { getUsers, createUser } from '../api/index'
import { DEPARTMENTS } from '../utils/departments'

const users = ref([])
const form = ref({ username: '', password: '', department: '公共', role: 'user' })
const message = ref('')

async function load() {
  try {
    users.value = await getUsers()
  } catch (e) {
    message.value = e.response?.data?.detail || '加载失败'
  }
}

async function submit() {
  if (!form.value.username || !form.value.password) {
    message.value = '请填写用户名和密码'
    return
  }
  try {
    await createUser({ ...form.value })
    message.value = `已创建：${form.value.username}`
    form.value = { username: '', password: '', department: '公共', role: 'user' }
    await load()
  } catch (e) {
    message.value = e.response?.data?.detail || '创建失败'
  }
}

onMounted(load)
</script>

<template>
  <section class="panel">
    <header class="panel-head">
      <h3 class="panel-label">用户管理</h3>
      <span class="panel-meta">{{ users.length }} 人</span>
    </header>
    <div class="panel-body">
      <div class="user-list">
        <div v-for="u in users" :key="u.id" class="user-item">
          <span class="user-name">{{ u.username }}</span>
          <span class="user-meta">{{ u.department }} · {{ u.role }}</span>
        </div>
      </div>

      <div class="user-form">
        <input v-model="form.username" placeholder="用户名" />
        <input v-model="form.password" type="password" placeholder="密码" />
        <div class="user-form-row">
          <select v-model="form.department">
            <option v-for="d in DEPARTMENTS" :key="d" :value="d">{{ d }}</option>
          </select>
          <select v-model="form.role">
            <option value="user">user</option>
            <option value="admin">admin</option>
          </select>
        </div>
        <button class="btn" @click="submit">创建用户</button>
      </div>

      <p v-if="message" class="doc-msg">{{ message }}</p>
    </div>
  </section>
</template>
