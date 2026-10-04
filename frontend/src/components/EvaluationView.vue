<script setup>
import { ref, watch } from 'vue'
import { getEvaluation } from '../api'
import { t } from '../i18n'

const props = defineProps({ open: { type: Boolean, default: false } })
const emit = defineEmits(['close'])

const loading = ref(false)
const data = ref(null)
const error = ref('')
const ks = [1, 3, 5, 10]

watch(() => props.open, async (v) => {
  if (!v) return
  loading.value = true
  error.value = ''
  data.value = null
  try {
    data.value = await getEvaluation()
  } catch (e) {
    error.value = e.response?.data?.detail || 'Failed to load report'
  }
  loading.value = false
})

function fmt(v) {
  return v == null ? '—' : typeof v === 'number' ? v.toFixed(3) : v
}
</script>

<template>
  <div v-if="open" class="modal-mask" @click.self="emit('close')">
    <div class="modal">
      <header class="modal-head">
        <h2 class="modal-title">{{ t('evaluation.title') }}</h2>
        <button class="modal-close" @click="emit('close')">×</button>
      </header>
      <div class="modal-body">
        <p v-if="loading" class="note">{{ t('chat.searching') }}</p>
        <p v-else-if="error" class="note">{{ error }}</p>
        <template v-else-if="data && data.available">
          <div class="eval-meta">
            <div class="eval-cell">
              <span class="eval-num">{{ data.report.corpus_size }}</span>
              <span class="eval-key">{{ t('evaluation.corpusSize') }}</span>
            </div>
            <div class="eval-cell">
              <span class="eval-num">{{ data.report.question_count }}</span>
              <span class="eval-key">{{ t('evaluation.questionCount') }}</span>
            </div>
          </div>
          <table class="eval-table">
            <thead>
              <tr>
                <th>{{ t('evaluation.strategy') }}</th>
                <th v-for="k in ks" :key="'r' + k">R@{{ k }}</th>
                <th v-for="k in ks" :key="'m' + k">MRR@{{ k }}</th>
              </tr>
            </thead>
            <tbody>
              <tr v-for="(metrics, name) in data.report.summary" :key="name">
                <td class="eval-name">{{ name }}</td>
                <td v-for="k in ks" :key="'r' + k">{{ fmt(metrics['recall@' + k]) }}</td>
                <td v-for="k in ks" :key="'m' + k">{{ fmt(metrics['mrr@' + k]) }}</td>
              </tr>
            </tbody>
          </table>
        </template>
        <p v-else class="note">{{ t('evaluation.unavailable') }}</p>
      </div>
    </div>
  </div>
</template>
