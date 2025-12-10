import os
import requests
from urllib.parse import urlparse

def get_documents(url):
    print(f"🕵️ Investigando a página: {url} ...")

    # 1. Prepara a pasta
    pasta = os.path.join(".", "docs_uploader", "documents")
    if not os.path.exists(pasta):
        os.makedirs(pasta, exist_ok=True)
        print(f"📁 Estrutura de pastas criada: {pasta}")
    else:
        print(f"📂 Usando pasta existente: {pasta}")

    nome_arquivo = os.path.basename(urlparse(url).path)
    if not nome_arquivo.lower().endswith('.pdf'):
        nome_arquivo = "documento.pdf"
        
    caminho_completo = os.path.join(pasta, nome_arquivo)

    print(f"⬇️ Baixando arquivo...")

    try:
        # User-Agent robusto para evitar bloqueios simples
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
        }
        
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()

        # --- VERIFICAÇÃO DE SEGURANÇA ---
        
        # 1. Verifica o Tipo de Conteúdo (Content-Type)
        tipo_conteudo = response.headers.get('Content-Type', '').lower()
        if 'pdf' not in tipo_conteudo and 'application/octet-stream' not in tipo_conteudo:
            print(f"❌ ERRO: O link retornou um tipo '{tipo_conteudo}' em vez de PDF.")
            print("   Provavelmente é uma página HTML. Verifique se a URL é o link DIRETO do arquivo.")
            return

        # 2. Verifica a Assinatura do Arquivo (Magic Number)
        # Todo PDF real começa com os bytes: %PDF
        if not response.content.startswith(b'%PDF'):
            print("❌ ERRO: O arquivo baixado não começa com a assinatura de um PDF.")
            print("   Conteúdo inicial baixado (primeiros 100 caracteres):")
            print(f"   {response.content[:100]}") # Mostra o que veio errado
            return
            
        # -------------------------------

        with open(caminho_completo, 'wb') as f:
            f.write(response.content)

        print(f"✅ Sucesso! PDF válido salvo em:\n   {os.path.abspath(caminho_completo)}")

    except Exception as e:
        print(f"❌ Erro: {e}")