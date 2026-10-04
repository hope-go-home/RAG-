export const ARTICLE_TYPES = [
  { name: 'article', labelKey: 'type.article', color: '#2b4bf2' },
  { name: 'feature_request', labelKey: 'type.feature_request', color: '#6a7fff' },
  { name: 'known_issue', labelKey: 'type.known_issue', color: '#ea5a2b' },
]

const COLOR_MAP = Object.fromEntries(ARTICLE_TYPES.map((t) => [t.name, t.color]))

export function typeColor(name) {
  return COLOR_MAP[name] || '#7b89a6'
}
