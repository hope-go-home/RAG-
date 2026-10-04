<script setup>
import { ref, onMounted } from 'vue'
import { getDocuments, deleteDocument, reindexDocument } from '../api'
import { t } from '../i18n'

const docs = ref([])
const loading = ref(false)
const message = ref('')

async function load() {
  loading.value = true
  try {
    docs.value = await getDocuments(50)
    message.value = ''
  } catch (e) {
    message.value = e.response?.data?.detail || 'Load failed'
  } finally {
    loading.value = false
  }
}

async function handleDelete(d) {
  if (!confirm(t('docs.confirm').replace('{name}', d.title || d.source))) return
  try {
    await deleteDocument(d.id)
    message.value = t('docs.deleted').replace('{name}', d.source)
    await load()
  } catch (e) {
    message.value = e.response?.data?.detail || 'Delete failed'
  }
}

async function handleReindex(d) {
  try {
    const r = await reindexDocument(d.id)
    message.value = t('docs.reindexed')
      .replace('{name}', d.source).replace('{n}', r.chunk_count).replace('{v}', r.version)
    await load()
  } catch (e) {
    message.value = e.response?.data?.detail || 'Reindex failed'
  }
}

onMounted(load)
</script>

<template>
  <section class="panel">
    <header class="panel-head">
      <h3 class="panel-label">{{ t('docs.title') }}</h3>
      <button class="panel-meta doc-refresh" @click="load">{{ t('docs.refresh') }}</button>
    </header>
    <div class="panel-body">
      <div v-if="!docs.length" class="note">{{ t('docs.empty') }}</div>
      <div v-else class="doc-list">
        <div v-for="d in docs" :key="d.id" class="doc-item">
          <div class="doc-main">
            <a
              v-if="d.url"
              class="doc-name doc-link"
              :href="d.url"
              target="_blank"
              rel="noopener noreferrer"
              :title="d.source"
            >{{ d.title || d.source }} <span class="src-ext">↗</span></a>
            <span v-else class="doc-name" :title="d.source">{{ d.source }}</span>
            <span class="doc-meta">{{ d.chunk_count }} {{ t('docs.chunks') }} · v{{ d.version }}</span>
          </div>
          <div class="doc-actions">
            <button :title="t('docs.reindex')" @click="handleReindex(d)">{{ t('docs.reindex') }}</button>
            <button :title="t('docs.delete')" class="doc-del" @click="handleDelete(d)">{{ t('docs.delete') }}</button>
          </div>
        </div>
      </div>
      <p v-if="message" class="doc-msg">{{ message }}</p>
    </div>
  </section>
</template>
