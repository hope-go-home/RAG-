export const DEPARTMENTS = ['HR', '财务', 'IT', '技术', '公共']

export const DEFAULT_DEPARTMENT = 'HR'

export function deptHint(dept) {
  return dept === '公共' ? '仅检索公共文档' : `仅检索「${dept}」+ 公共文档`
}
