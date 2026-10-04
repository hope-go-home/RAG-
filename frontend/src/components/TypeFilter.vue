<script setup>
import { useChatStore } from '../stores/chat'
import { ARTICLE_TYPES, typeColor } from '../utils/articleTypes'
import { t } from '../i18n'

const store = useChatStore()
</script>

<template>
  <section class="panel">
    <header class="panel-head">
      <h3 class="panel-label">{{ t('filter.label') }}</h3>
      <span class="panel-meta">{{ store.docType ? t('type.' + store.docType) : t('filter.all') }}</span>
    </header>
    <div class="panel-body" style="padding: 8px">
      <div class="type-grid">
        <button
          :class="['type-btn', { active: !store.docType }]"
          @click="store.docType = ''"
        >
          <span class="type-dot" style="background:#7b89a6" />{{ t('filter.all') }}
        </button>
        <button
          v-for="ty in ARTICLE_TYPES"
          :key="ty.name"
          :class="['type-btn', { active: store.docType === ty.name }]"
          @click="store.docType = store.docType === ty.name ? '' : ty.name"
        >
          <span class="type-dot" :style="{ background: typeColor(ty.name) }" />{{ t(ty.labelKey) }}
        </button>
      </div>
    </div>
  </section>
</template>
