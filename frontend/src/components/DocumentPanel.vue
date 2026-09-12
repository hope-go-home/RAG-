<script setup>
import { ref, onMounted } from 'vue'
import { getDocuments, deleteDocument, reindexDocument } from '../api/index'

const docs = ref([])
const loading = ref(false)
const message = ref('')

async function load() {
  loading.value = true
  try {
    docs.value = await getDocuments()
    message.value = ''
  } catch (e) {
    message.value = e.response?.data?.detail || '加载失败'
  } finally {
    loading.value = false
  }
}

async function handleDelete(d) {
  if (!confirm(`确认删除《${d.source}》？`)) return
  try {
    await deleteDocument(d.id)
    message.value = `已删除：${d.source}`
    await load()
  } catch (e) {
    message.value = e.response?.data?.detail || '删除失败'
  }
}

async function handleReindex(d) {
  try {
    const r = await reindexDocument(d.id)
    message.value = `已重建：${d.source}（${r.chunk_count} 块，v${r.version}）`
    await load()
  } catch (e) {
    message.value = e.response?.data?.detail || '重建失败'
  }
}

onMounted(load)
</script>

<template>
  <section class="panel">
    <header class="panel-head">
      <h3 class="panel-label">文档管理</h3>
      <button class="panel-meta doc-refresh" @click="load">刷新</button>
    </header>
    <div class="panel-body">
      <div v-if="!docs.length" class="note">暂无已入库文档</div>
      <div v-else class="doc-list">
        <div v-for="d in docs" :key="d.id" class="doc-item">
          <div class="doc-main">
            <span class="doc-name" :title="d.source">{{ d.source }}</span>
            <span class="doc-meta">{{ d.department }} · {{ d.chunk_count }} 块 · v{{ d.version }}</span>
          </div>
          <div class="doc-actions">
            <button title="重建索引" @click="handleReindex(d)">重建</button>
            <button title="删除文档" class="doc-del" @click="handleDelete(d)">删除</button>
          </div>
        </div>
      </div>
      <p v-if="message" class="doc-msg">{{ message }}</p>
    </div>
  </section>
</template>
