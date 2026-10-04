<script setup>
import { useChatStore } from '../stores/chat'
import { t } from '../i18n'

const store = useChatStore()
</script>

<template>
  <section class="panel">
    <header class="panel-head">
      <h3 class="panel-label">{{ t('panels.stats') }}</h3>
      <span class="panel-meta">dense · sparse · rrf · rerank</span>
    </header>
    <div class="panel-body">
      <div v-if="store.stats" class="readout">
        <div class="readout-cell">
          <div class="readout-num">{{ store.stats.dense_hits ?? '—' }}</div>
          <div class="readout-key">{{ t('stats.dense') }}</div>
        </div>
        <div class="readout-cell">
          <div class="readout-num">{{ store.stats.sparse_hits ?? '—' }}</div>
          <div class="readout-key">{{ t('stats.sparse') }}</div>
        </div>
        <div class="readout-cell">
          <div class="readout-num">{{ store.stats.fused_count ?? '—' }}</div>
          <div class="readout-key">{{ t('stats.fused') }}</div>
        </div>
        <div class="readout-cell">
          <div class="readout-num">{{ store.stats.reranked_count ?? '—' }}</div>
          <div class="readout-key">{{ t('stats.reranked') }}</div>
        </div>
      </div>
      <div v-else class="note">{{ t('stats.waiting') }}</div>
    </div>
  </section>
</template>
