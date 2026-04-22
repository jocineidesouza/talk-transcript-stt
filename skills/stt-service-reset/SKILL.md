---
name: stt-service-reset
description: Limpar runtime local do serviço STT em Docker Compose e subir novamente sem baixar modelos. Use quando precisar resetar `sqlite/spool`, parar o serviço, rebuildar imagem e subir `stt` preservando `stt/models`.
---

# STT Service Reset

Executar reset completo do serviço STT local sem apagar modelos.

## Fluxo

1. Ir para a raiz do projeto onde existe `docker-compose.yml`.
2. Executar o script:

```powershell
powershell -ExecutionPolicy Bypass -File .\skills\stt-service-reset\scripts\reset-stt-service.ps1
```

## O que o script faz

1. `docker compose stop stt`
2. Remove arquivos SQLite em `stt/data` (`queue.db`, `queue.db-shm`, `queue.db-wal`)
3. Limpa conteúdo de `stt/data/spool` (mantém a pasta)
4. `docker compose build stt`
5. `docker compose up -d stt`
6. Mostra status com `docker compose ps`

## Restrições de segurança

- Nunca remover `stt/models`.
- Falhar se `docker-compose.yml` não existir no diretório informado.
- Operar apenas dentro do projeto informado por `-ProjectRoot` (ou diretório atual).

## Parâmetros opcionais

- `-ProjectRoot <path>`: raiz do projeto (padrão: diretório atual).

