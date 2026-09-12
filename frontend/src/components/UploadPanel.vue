<script setup>
import { ref } from 'vue'
import { uploadFiles } from '../api/index'
import { DOC_TYPES } from '../utils/docTypes'
import { DEPARTMENTS, DEFAULT_DEPARTMENT } from '../utils/departments'

const files = ref([])
const docType = ref(DOC_TYPES[0].name)
const department = ref(DEFAULT_DEPARTMENT)
const uploading = ref(false)
const results = ref([])
const over = ref(false)
const fileInput = ref(null)

function onFileChange(e) {
  files.value = [...e.target.files]
}

function onDrop(e) {
  e.preventDefault()
  over.value = false
  files.value = [...e.dataTransfer.files]
}

function onDragOver(e) {
  e.preventDefault()
  over.value = true
}

function onDragLeave() {
  over.value = false
}

async function handleUpload() {
  if (!files.value.length) return
  uploading.value = true
  results.value = []
  try {
    const res = await uploadFiles(files.value, docType.value, department.value)
    results.value = res.results || []
    files.value = []
    if (fileInput.value) fileInput.value.value = ''
  } catch (e) {
    results.value = [{
      file: '上传',
      status: 'failed',
      message: e.response?.data?.detail || e.message || '上传失败',
    }]
  } finally {
    uploading.value = false
  }
}
</script>

<template>
  <section class="panel">
    <header class="panel-head">
      <h3 class="panel-label">文档入库</h3>
      <span class="panel-meta">pdf · docx · md · xlsx · 图片</span>
    </header>
    <div class="panel-body">
      <div
        class="drop"
        :class="{ over }"
        @drop="onDrop"
        @dragover="onDragOver"
        @dragleave="onDragLeave"
        @click="fileInput.click()"
      >
        <div class="drop-plus">＋</div>
        <div class="drop-main">{{ files.length ? `已选 ${files.length} 份` : '点击或拖入文件' }}</div>
        <div class="drop-hint">PDF / WORD / TXT / MD / EXCEL / 图片(OCR)</div>
        <input
          ref="fileInput"
          type="file"
          multiple
          accept=".pdf,.docx,.txt,.md,.xlsx,.png,.jpg,.jpeg"
          style="display: none"
          @change="onFileChange"
        />
      </div>

      <div class="type-grid">
        <button
          v-for="d in DOC_TYPES"
          :key="d.name"
          :class="['type-btn', { active: docType === d.name }]"
          @click="docType = d.name"
        >
          <span class="type-dot" :style="{ background: d.color }" />{{ d.short }}
        </button>
      </div>

      <label class="dept dept-block">
        <span class="dept-label">归属部门</span>
        <select v-model="department" class="dept-select">
          <option v-for="d in DEPARTMENTS" :key="d" :value="d">{{ d }}</option>
        </select>
      </label>

      <div class="field-row">
        <button class="btn" style="flex: 1" :disabled="!files.length || uploading" @click="handleUpload">
          {{ uploading ? '入库中' : '入库' }}
        </button>
      </div>

      <div v-if="results.length" class="results">
        <div v-for="(r, i) in results" :key="i" :class="['result', r.status]">
          <span v-if="r.ocr" class="result-ocr">OCR</span>{{ r.file }} · {{ r.message }}
        </div>
      </div>
    </div>
  </section>
</template>
