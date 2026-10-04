<script setup>
import { locale, setLocale } from '../i18n'
</script>

<template>
  <div class="lang" role="group" aria-label="Language">
    <button
      :class="['lang-btn', { active: locale === 'zh' }]"
      :aria-pressed="locale === 'zh'"
      @click="setLocale('zh')"
    >中</button>
    <button
      :class="['lang-btn', { active: locale === 'en' }]"
      :aria-pressed="locale === 'en'"
      @click="setLocale('en')"
    >EN</button>
  </div>
</template>
