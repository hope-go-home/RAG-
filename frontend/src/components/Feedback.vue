<script setup>
import { ref } from 'vue'
import { sendFeedback } from '../api'
import { t } from '../i18n'

const props = defineProps({
  question: { type: String, default: '' },
  answer: { type: String, default: '' },
  sessionId: { type: String, default: '' },
})

const given = ref('')
const sending = ref(false)

async function vote(rating) {
  if (given.value || sending.value) return
  sending.value = true
  try {
    await sendFeedback({
      rating,
      question: props.question,
      answer: props.answer,
      session_id: props.sessionId,
    })
  } catch {
    // 反馈失败不影响使用
  }
  given.value = rating
  sending.value = false
}
</script>

<template>
  <div class="fb">
    <span v-if="given" class="fb-thanks">{{ t('feedback.thanks') }}</span>
    <template v-else>
      <button class="fb-btn" :disabled="sending" :title="t('feedback.up')" @click="vote('up')">👍</button>
      <button class="fb-btn" :disabled="sending" :title="t('feedback.down')" @click="vote('down')">👎</button>
    </template>
  </div>
</template>
