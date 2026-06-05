import { useState } from 'react'
import { api } from './services/api'

function App() {
  const [nome, setNome] = useState('')
  const [foto, setFoto] = useState<File | null>(null)
  const [loading, setLoading] = useState(false)
  const [mensagem, setMensagem] = useState('')

  // O React.SyntheticEvent é o "pai" de todos os eventos e nunca fica obsoleto
  const handleSubmit = async (e: React.SyntheticEvent) => {
    e.preventDefault() 

    if (!nome || !foto) {
      setMensagem("Por favor, preencha o nome e selecione uma foto.")
      return
    }

    setLoading(true)
    setMensagem("Enviando para o servidor...")

    try {
      const resultado = await api.cadastrarUsuario(nome, foto)
      
      // 1. Isso vai imprimir no painel do navegador exatamente o que o Python devolveu:
      console.log("Resposta do Backend:", resultado)

      // 2. Você pode ajustar a linha abaixo com base no que você retorna no seu main.py
      // Se o seu Python retorna {"message": "..."} use resultado.message
      // Se retorna {"nome": "..."} use resultado.nome
      const textoResposta = resultado.message || resultado.nome || JSON.stringify(resultado)

      setMensagem(`Sucesso! Usuário cadastrado: ${textoResposta}`)
      
      setNome('')
      setFoto(null)
    } catch (error: any) {
      setMensagem(`Erro: ${error.message}`)
    } finally {
      setLoading(false)
    }
  }

  return (
    <div style={{ maxWidth: '500px', margin: '40px auto', fontFamily: 'system-ui, sans-serif' }}>
      <h1>MVP FaceRECOG</h1>
      <p>Envie uma foto para alimentar o banco de dados.</p>

      <form onSubmit={handleSubmit} style={{ display: 'flex', flexDirection: 'column', gap: '15px' }}>
        
        <div style={{ display: 'flex', flexDirection: 'column' }}>
          <label htmlFor="nome">Nome do Usuário:</label>
          <input
            id="nome"
            type="text"
            value={nome}
            // TypeScript infere o tipo automaticamente aqui, sem dar warning!
            onChange={(e) => setNome(e.target.value)}
            placeholder="Ex: Jhonson"
            style={{ padding: '8px', fontSize: '16px' }}
          />
        </div>

        <div style={{ display: 'flex', flexDirection: 'column' }}>
          <label htmlFor="foto">Selecione uma imagem:</label>
          <input
            id="foto"
            type="file"
            accept="image/*"
            // Ao invés de uma função separada, usamos o cast direto no target do DOM
            onChange={(e) => {
              const target = e.target as HTMLInputElement;
              if (target.files && target.files.length > 0) {
                setFoto(target.files[0])
              }
            }}
            style={{ padding: '8px', border: '1px solid #ccc' }}
          />
        </div>

        <button 
          type="submit" 
          disabled={loading}
          style={{ 
            padding: '12px', 
            fontSize: '16px', 
            backgroundColor: loading ? '#ccc' : '#0066cc', 
            color: 'white', 
            border: 'none', 
            cursor: loading ? 'not-allowed' : 'pointer' 
          }}
        >
          {loading ? 'Processando imagem...' : 'Cadastrar Rosto'}
        </button>
      </form>

      {mensagem && (
        <div style={{ marginTop: '20px', padding: '15px', backgroundColor: '#f5f5f5', borderRadius: '5px' }}>
          {mensagem}
        </div>
      )}
    </div>
  )
}

export default App