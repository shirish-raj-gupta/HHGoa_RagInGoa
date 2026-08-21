// Where the backend lives.
//
// This page is served from a Hugging Face Static Space, which can host files
// but cannot run Python - the account tier gates Docker and ZeroGPU Spaces.
// So the API runs elsewhere (a local container behind a Cloudflare Tunnel) and
// this points at it.
//
// Override without redeploying by appending ?api=https://your-backend to the
// URL; it is remembered in localStorage.
window.RAG_API = "";   // set by scripts/deploy_static_space.py at deploy time
