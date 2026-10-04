<script setup>
import { useChatStore } from '../stores/chat'
import { t, locale } from '../i18n'

const store = useChatStore()

const NODE_LABELS = {
  query_analysis: { zh: '意图分析', en: 'Intent analysis' },
  retrieve: { zh: '混合检索', en: 'Hybrid retrieval' },
  rewrite: { zh: '查询改写', en: 'Query rewrite' },
  generate: { zh: '生成回答', en: 'Generate' },
  direct_answer: { zh: '直接回答', en: 'Direct answer' },
}

function label(node) {
  const m = NODE_LABELS[node]
  if (m) return m[locale.value] || m.zh
  return node || t('panels.thinking')
}
</script>

<template>
  <section class="panel">
    <header class="panel-head">
      <h3 class="panel-label">{{ t('panels.thinking') }}</h3>
      <span v-if="store.thinking.length" class="panel-meta">{{ store.thinking.length }}</span>
    </header>
    <div class="panel-body">
      <div v-if="store.thinking.length" class="think">
        <div v-for="(step, i) in store.thinking" :key="i" class="think-row">
          <span class="think-num">{{ String(i + 1).padStart(2, '0') }}</span>
          <div class="think-content">
            <div class="think-name">{{ label(step.node) }}</div>
            <div v-if="step.info" class="think-detail">{{ step.info }}</div>
          </div>
        </div>
      </div>
      <div v-else class="note">{{ t('stats.waiting') }}</div>
    </div>
  </section>
</template>
