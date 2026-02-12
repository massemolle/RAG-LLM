# URLs – Quick reference

Use these to open the app and Langfuse. No code changes; same setup as when it was working.

---

## Your app (Streamlit)

| What | URL |
|------|-----|
| **RAG + Guardrails UI** | **http://localhost:8502** |

Start the app (from the project folder):

```bash
streamlit run streamv3.py --server.port 8502 --server.address localhost
```

If you need it on all interfaces:

```bash
streamlit run streamv3.py --server.port 8502 --server.address 0.0.0.0
```

---

## Langfuse (traces / observability)

| What | URL |
|------|-----|
| **Langfuse dashboard** | **https://cloud.langfuse.com** |
| **OTLP traces endpoint** (used by the app) | `https://cloud.langfuse.com/api/public/otel/v1/traces` |

You only need to open the dashboard in the browser. The app sends traces to the OTLP endpoint using the keys in `.env`.

---

## Env vars (from your `.env`)

The app reads:

- `LANGFUSE_PUBLIC_KEY` → from `.env`
- `LANGFUSE_SECRET_KEY` → from `.env`
- `LANGFUSE_BASE_URL` or `LANGFUSE_HOST` → default `https://cloud.langfuse.com`

So in practice:

1. **App**: open **http://localhost:8502** after running `streamlit run streamv3.py ...`.
2. **Langfuse**: open **https://cloud.langfuse.com** and log in; traces appear there when the app runs and sends data.

Nothing else needs to change for “localhost + Langfuse” to work again; the only requirement is that the environment allows the app to bind to a port and create sockets (as when it worked before).
