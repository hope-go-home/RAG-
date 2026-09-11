<script setup>
import { useChatStore } from '../stores/chat'

const store = useChatStore()
</script>

<template>
  <section class="panel">
    <header class="panel-head">
      <h3 class="panel-label">检索统计</h3>
      <span class="panel-meta">dense / sparse / rrf / rerank</span>
    </header>
    <div class="panel-body" style="padding: 7px 8px">
      <div v-if="store.stats" class="readout">
        <div class="readout-cell">
          <div class="readout-num">{{ store.stats.dense_hits ?? '—' }}</div>
          <div class="readout-key">向量命中</div>
        </div>
        <div class="readout-cell">
          <div class="readout-num">{{ store.stats.sparse_hits ?? '—' }}</div>
          <div class="readout-key">关键词命中</div>
        </div>
        <div class="readout-cell">
          <div class="readout-num">{{ store.stats.fused_count ?? '—' }}</div>
          <div class="readout-key">RRF 融合</div>
        </div>
        <div class="readout-cell">
          <div class="readout-num">{{ store.stats.reranked_count ?? '—' }}</div>
          <div class="readout-key">重排精选</div>
        </div>
      </div>
      <div v-else class="note">等待检索数据</div>
    </div>
  </section>
</template>
