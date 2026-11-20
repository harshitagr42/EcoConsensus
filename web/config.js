// The only setting of the website: where the backend (Hugging Face Space) is.
// `python scripts/hf_space.py status` prints both URLs. For local testing open index.html?api=http://localhost:7860
window.ECO_CONFIG = {
  API_BASE_URL: "https://YOUR-HF-USERNAME-ecoconsensus-api.hf.space",
  SPACE_PAGE_URL: "https://huggingface.co/spaces/YOUR-HF-USERNAME/ecoconsensus-api",
};
