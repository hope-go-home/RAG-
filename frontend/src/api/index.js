import axios from 'axios'
import { useAuthStore } from '../stores/auth'

const http = axios.create({
  baseURL: '',
  timeout: 300000,
})

http.interceptors.request.use((config) => {
  const auth = useAuthStore()
  if (auth.token) config.headers.Authorization = `Bearer ${auth.token}`
  return config
})

http.interceptors.response.use(
  (response) => response,
  (error) => {
    if (error.response?.status === 401) {
      const auth = useAuthStore()
      auth.logout()
    }
    return Promise.reject(error)
  },
)

export async function login(username, password) {
  const { data } = await http.post('/auth/login', { username, password })
  return data
}

export async function register(username, password, department) {
  const { data } = await http.post('/auth/register', { username, password, department })
  return data
}

export async function getUsers() {
  const { data } = await http.get('/auth/users')
  return data
}

export async function createUser(payload) {
  const { data } = await http.post('/auth/users', payload)
  return data
}

export async function getMe() {
  const { data } = await http.get('/auth/me')
  return data
}

export async function uploadFiles(files, docType, department) {
  const formData = new FormData()
  for (const file of files) {
    formData.append('files', file)
  }
  formData.append('doc_type', docType)
  if (department) formData.append('department', department)
  const { data } = await http.post('/upload', formData, {
    headers: { 'Content-Type': 'multipart/form-data' },
  })
  return data
}

export async function getHistory(sessionId) {
  const { data } = await http.get(`/history/${sessionId}`)
  return data
}

export async function getSessions(limit = 50) {
  const { data } = await http.get('/sessions', { params: { limit } })
  return data
}

export async function deleteSession(sessionId) {
  const { data } = await http.delete(`/sessions/${sessionId}`)
  return data
}

export async function getDocuments(limit = 200) {
  const { data } = await http.get('/documents', { params: { limit } })
  return data
}

export async function deleteDocument(id) {
  const { data } = await http.delete(`/documents/${id}`)
  return data
}

export async function reindexDocument(id) {
  const { data } = await http.post(`/documents/${id}/reindex`)
  return data
}

export default http
