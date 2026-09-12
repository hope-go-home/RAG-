<script setup>
import { computed } from 'vue'
import { marked } from 'marked'
import DOMPurify from 'dompurify'

const props = defineProps({
  role: { type: String, required: true },
  content: { type: String, default: '' },
  streaming: { type: Boolean, default: false },
})

const rendered = computed(() => {
  if (!props.content) return ''
  // 知识库文档内容不可信，渲染前必须消毒，防止 XSS
  return DOMPurify.sanitize(marked.parse(props.content, { breaks: true, gfm: true }))
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
