// frontend/src/services/api.ts

// Puxa a URL do .env ou usa o localhost como fallback de segurança
const BASE_URL = import.meta.env.VITE_API_URL || "http://localhost:8000";

export const api = {
  /**
   * Zera o banco de dados (A rota que criamos mais cedo)
   */
  deletarTodosUsuarios: async () => {
    try {
      const response = await fetch(`${BASE_URL}/deletar-todos`, {
        method: "DELETE",
      });
      
      if (!response.ok) {
        throw new Error("Erro ao limpar o banco de dados.");
      }
      
      return await response.json();
    } catch (error) {
      console.error("Erro na API:", error);
      throw error;
    }
  },

  // Futuramente, colocaremos aqui a função de enviarFotosCamera()

  /**
   * Envia uma foto e um nome para a rota /cadastrar do FastAPI
   */
  cadastrarUsuario: async (nome: string, file: File) => {
    // FormData é obrigatório para enviar arquivos no JavaScript
    const formData = new FormData();
    formData.append("nome", nome);
    formData.append("file", file);

    try {
      const response = await fetch(`${BASE_URL}/cadastrar`, {
        method: "POST",
        body: formData, 
        // Nota: NÃO coloque o cabeçalho 'Content-Type' aqui! 
        // O navegador calcula automaticamente o 'boundary' correto para FormData.
      });
      
      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || "Erro desconhecido ao cadastrar.");
      }
      
      return await response.json();
    } catch (error) {
      console.error("Erro no cadastro:", error);
      throw error;
    }
    
  },
  reconhecerUsuario: async (file: File) => {
    const formData = new FormData();
    // Verifique se o seu backend espera o campo com o nome "file" ou "foto"
    formData.append("file", file); 

    try {
      const response = await fetch(`${BASE_URL}/reconhecer`, {
        method: "POST",
        body: formData,
      });
      
      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || "Erro ao tentar reconhecer o rosto.");
      }
      
      return await response.json();
    } catch (error) {
      console.error("Erro no reconhecimento:", error);
      throw error;
    }
  }
  
};

