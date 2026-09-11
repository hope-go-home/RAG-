<script setup>
import { computed } from 'vue'
import { marked } from 'marked'

const props = defineProps({
  role: { type: String, required: true },
  content: { type: String, default: '' },
  streaming: { type: Boolean, default: false },
})

const rendered = computed(() => {
  if (!props.content) return ''
  return marked.parse(props.content, { breaks: true, gfm: true })
})
</script>

<template>
  <div :class="['msg', role]">
    <div class="msg-tag">{{ role === 'user' ? '问' : '答' }}</div>
    <div :class="['msg-body', { streaming }]">
      <div v-if="content" class="md" v-html="rendered" />
      <span v-else-if="role === 'assistant'" class="pending">生成中</span>
    </div>
  </div>
</template>
