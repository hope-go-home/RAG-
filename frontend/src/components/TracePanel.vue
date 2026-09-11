<script setup>
import { useChatStore } from '../stores/chat'
import { docColor, docShort } from '../utils/docTypes'

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
      <h3 class="panel-label">检索流转</h3>
      <span v-if="store.sources.length" class="panel-meta">{{ store.sources.length }} 条来源</span>
    </header>
    <div class="panel-body">
      <div v-if="store.sources.length" class="trace">
        <article
          v-for="(src, i) in store.sources"
          :key="i"
          class="trace-card"
          :style="{ borderLeftColor: docColor(src.doc_type) }"
        >
          <div class="trace-head">
            <span class="trace-rank">{{ String(i + 1).padStart(2, '0') }}</span>
            <span
              class="trace-type"
              :style="{ background: docColor(src.doc_type) }"
            >{{ docShort(src.doc_type) }}</span>
            <span v-if="src.dense_rank" class="trace-chan">D#{{ src.dense_rank }}</span>
            <span v-if="src.sparse_rank" class="trace-chan">S#{{ src.sparse_rank }}</span>
            <span class="trace-arrow">→</span>
            <span v-if="fmt(src.rrf_score, 4)" class="trace-metric">RRF {{ fmt(src.rrf_score, 4) }}</span>
            <span v-if="fmt(src.rerank_score)" class="trace-metric">↑ {{ fmt(src.rerank_score) }}</span>
          </div>
          <p class="trace-text">{{ src.text || '' }}</p>
        </article>
      </div>
      <div v-else class="note">等待检索结果</div>
    </div>
  </section>
</template>
