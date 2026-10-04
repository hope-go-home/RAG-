import { ref } from 'vue'
import zh from './zh'
import en from './en'

const messages = { zh, en }
const STORAGE_KEY = 'kb_lang'

function readLang() {
  try {
    return localStorage.getItem(STORAGE_KEY)
  } catch {
    return null
  }
}

export const locale = ref(readLang() || 'zh')
export const availableLocales = ['zh', 'en']

export function setLocale(l) {
  if (!messages[l]) return
  locale.value = l
  try {
    localStorage.setItem(STORAGE_KEY, l)
    document.documentElement.setAttribute('lang', l === 'zh' ? 'zh-CN' : 'en')
  } catch {
    // ignore
  }
}

function lookup(obj, path) {
  return path.split('.').reduce((o, k) => (o && o[k] != null ? o[k] : undefined), obj)
}

export function t(key) {
  const table = messages[locale.value] || messages.zh
  const val = lookup(table, key)
  if (val != null) return val
  const fallback = lookup(messages.zh, key)
  return fallback != null ? fallback : key
}

export default { locale, setLocale, t, availableLocales }
