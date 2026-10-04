<script setup>
import { ref } from 'vue'
import { uploadFiles } from '../api'
import { ARTICLE_TYPES, typeColor } from '../utils/articleTypes'
import { t } from '../i18n'

const files = ref([])
const docType = ref('article')
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
    const res = await uploadFiles(files.value, docType.value)
    results.value = res.results || []
    files.value = []
    if (fileInput.value) fileInput.value.value = ''
  } catch (e) {
    results.value = [{
      file: 'upload',
      status: 'failed',
      message: e.response?.data?.detail || e.message || t('upload.failed'),
    }]
  } finally {
    uploading.value = false
  }
}
</script>

<template>
  <section class="panel">
    <header class="panel-head">
      <h3 class="panel-label">{{ t('upload.title') }}</h3>
      <span class="panel-meta">{{ t('upload.formats') }}</span>
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
        <div class="drop-main">
          {{ files.length ? t('upload.selected').replace('{n}', files.length) : t('upload.drop') }}
        </div>
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
          v-for="ty in ARTICLE_TYPES"
          :key="ty.name"
          :class="['type-btn', { active: docType === ty.name }]"
          @click="docType = ty.name"
        >
          <span class="type-dot" :style="{ background: typeColor(ty.name) }" />{{ t(ty.labelKey) }}
        </button>
      </div>

      <div class="field-row">
        <button class="btn" style="flex: 1" :disabled="!files.length || uploading" @click="handleUpload">
          {{ uploading ? t('upload.uploading') : t('upload.submit') }}
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
