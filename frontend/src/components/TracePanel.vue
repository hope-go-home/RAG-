<script setup>
import { useChatStore } from '../stores/chat'
import { typeColor } from '../utils/articleTypes'
import { t } from '../i18n'

const store = useChatStore()

function fmt(v, digits = 3) {
  if (v === null || v === undefined || v === '') return null
  const n = Number(v)
  return Number.isNaN(n) ? null : n.toFixed(digits)
}
</script>

<template>
  <section class="panel">
    <header class="panel-head">
      <h3 class="panel-label">{{ t('panels.trace') }}</h3>
      <span v-if="store.sources.length" class="panel-meta">
        {{ store.sources.length }} {{ t('trace.sources') }}
      </span>
    </header>
    <div class="panel-body">
      <div v-if="store.sources.length" class="trace">
        <article
          v-for="(src, i) in store.sources"
          :key="i"
          class="trace-card"
          :style="{ borderLeftColor: typeColor(src.doc_type) }"
        >
          <div class="trace-head">
            <span class="trace-rank">{{ String(i + 1).padStart(2, '0') }}</span>
            <span class="trace-type" :style="{ background: typeColor(src.doc_type) }">
              {{ src.doc_type ? t('type.' + src.doc_type) : '—' }}
            </span>
            <span v-if="src.dense_rank" class="trace-chan">D#{{ src.dense_rank }}</span>
            <span v-if="src.sparse_rank" class="trace-chan sparse">S#{{ src.sparse_rank }}</span>
            <span class="trace-arrow">→</span>
            <span v-if="fmt(src.rrf_score, 4)" class="trace-metric">RRF {{ fmt(src.rrf_score, 4) }}</span>
            <span v-if="fmt(src.rerank_score)" class="trace-metric signal">↑ {{ fmt(src.rerank_score) }}</span>
          </div>
          <a
            v-if="src.url"
            class="trace-source"
            :href="src.url"
            target="_blank"
            rel="noopener noreferrer"
          >{{ src.title || src.source }} <span class="src-ext">↗</span></a>
          <p v-else-if="src.source" class="trace-source">{{ src.source }}</p>
          <p class="trace-text">{{ src.text || '' }}</p>
        </article>
      </div>
      <div v-else class="note">{{ t('trace.waiting') }}</div>
    </div>
  </section>
</template>
