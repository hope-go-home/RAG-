<script setup>
import { typeColor } from '../utils/articleTypes'
import { t } from '../i18n'

defineProps({
  item: { type: Object, required: true },
  index: { type: Number, default: 0 },
})
</script>

<template>
  <article class="src-card">
    <div class="src-top">
      <span class="src-rank">{{ String(index + 1).padStart(2, '0') }}</span>
      <span
        class="src-type"
        :style="{ background: typeColor(item.doc_type) }"
      >{{ item.doc_type ? t('type.' + item.doc_type) : '—' }}</span>
      <a
        v-if="item.url"
        class="src-title"
        :href="item.url"
        target="_blank"
        rel="noopener noreferrer"
      >{{ item.title || item.source }}<span class="src-ext">↗</span></a>
      <span v-else class="src-title">{{ item.title || item.source || '—' }}</span>
    </div>
    <p v-if="item.source" class="src-file">{{ item.source }}</p>
    <p class="src-snippet">{{ item.text || '' }}</p>
  </article>
</template>
