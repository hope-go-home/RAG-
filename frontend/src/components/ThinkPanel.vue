<script setup>
import { useChatStore } from '../stores/chat'

const store = useChatStore()

const NODE_LABELS = {
  query_analysis: '意图分析',
  retrieve: '混合检索',
  generate: '生成回答',
  direct_answer: '直接回答',
}

function label(node) {
  return NODE_LABELS[node] || node || '步骤'
}
</script>

<template>
  <section class="panel">
    <header class="panel-head">
      <h3 class="panel-label">思考过程</h3>
      <span v-if="store.thinking.length" class="panel-meta">{{ store.thinking.length }} 步</span>
    </header>
    <div class="panel-body">
      <div v-if="store.thinking.length" class="think">
        <div v-for="(step, i) in store.thinking" :key="i" class="think-row">
          <span class="think-num">{{ String(i + 1).padStart(2, '0') }}</span>
          <div>
            <div class="think-name">{{ label(step.node) }}</div>
            <div v-if="step.info" class="think-detail">{{ step.info }}</div>
          </div>
        </div>
      </div>
      <div v-else class="note">提问后显示推理链路</div>
    </div>
  </section>
</template>
