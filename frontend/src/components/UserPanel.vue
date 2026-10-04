<script setup>
import { ref, onMounted } from 'vue'
import { getUsers, createUser } from '../api'
import { t } from '../i18n'

const users = ref([])
const form = ref({ username: '', password: '', role: 'user' })
const message = ref('')

async function load() {
  try {
    users.value = await getUsers()
  } catch (e) {
    message.value = e.response?.data?.detail || 'Load failed'
  }
}

async function submit() {
  if (!form.value.username || !form.value.password) {
    message.value = t('users.fill')
    return
  }
  try {
    await createUser({ ...form.value })
    message.value = t('users.created').replace('{name}', form.value.username)
    form.value = { username: '', password: '', role: 'user' }
    await load()
  } catch (e) {
    message.value = e.response?.data?.detail || 'Create failed'
  }
}

onMounted(load)
</script>

<template>
  <section class="panel">
    <header class="panel-head">
      <h3 class="panel-label">{{ t('users.title') }}</h3>
      <span class="panel-meta">{{ t('users.count').replace('{n}', users.length) }}</span>
    </header>
    <div class="panel-body">
      <div class="user-list">
        <div v-for="u in users" :key="u.id" class="user-item">
          <span class="user-name">{{ u.username }}</span>
          <span class="user-meta">{{ u.role }}</span>
        </div>
      </div>

      <div class="user-form">
        <input v-model="form.username" :placeholder="t('users.username')" />
        <input v-model="form.password" type="password" :placeholder="t('users.password')" />
        <div class="user-form-row">
          <select v-model="form.role">
            <option value="user">{{ t('users.roleUser') }}</option>
            <option value="admin">{{ t('users.roleAdmin') }}</option>
          </select>
          <button class="btn" @click="submit">{{ t('users.create') }}</button>
        </div>
      </div>

      <p v-if="message" class="doc-msg">{{ message }}</p>
    </div>
  </section>
</template>
