import axios from 'axios'

const http = axios.create({
  baseURL: '',
  timeout: 300000,
})

export async function uploadFiles(files, docType) {
  const formData = new FormData()
  for (const file of files) {
    formData.append('files', file)
  }
  formData.append('doc_type', docType)
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

export default http
