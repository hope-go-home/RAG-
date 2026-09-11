export const DOC_TYPES = [
  { name: '规章制度', short: '规章', color: '#8f3a30' },
  { name: 'FAQ', short: 'FAQ', color: '#2f6b5c' },
  { name: '操作流程SOP', short: 'SOP', color: '#a8742a' },
  { name: '技术文档', short: '技术', color: '#35588a' },
  { name: '员工手册', short: '手册', color: '#5d4a78' },
  { name: '数据报表', short: '报表', color: '#4a6b35' },
]

const COLOR_MAP = Object.fromEntries(DOC_TYPES.map((d) => [d.name, d.color]))
const SHORT_MAP = Object.fromEntries(DOC_TYPES.map((d) => [d.name, d.short]))

export function docColor(name) {
  return COLOR_MAP[name] || '#8b887f'
}

export function docShort(name) {
  return SHORT_MAP[name] || name || '未知'
}
