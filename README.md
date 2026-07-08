# HDSP Group — sitio web

Sitio del grupo de investigación HDSP (UIS), construido con [Astro](https://astro.build/) y
[Tailwind CSS](https://tailwindcss.com/) y desplegado en GitHub Pages.

## Desarrollo

```bash
npm install
npm run dev      # servidor local de desarrollo
npm run build    # build de producción a ./dist
npm run preview  # previsualiza el build
```

## Publicaciones (actualización automática)

La página de publicaciones se genera a partir de `src/data/openalex-publications.json`.
Ese archivo lo produce el script [`scripts/fetch-openalex-publications.mjs`](scripts/fetch-openalex-publications.mjs),
que combina dos fuentes:

- **OpenAlex** (por DOI) — fuente principal de metadatos.
- **Google Scholar** — aporta las publicaciones que OpenAlex no tiene.

El workflow [`.github/workflows/update-publications.yml`](.github/workflows/update-publications.yml)
ejecuta ese script **cada lunes** (y también se puede lanzar manualmente desde
**Actions → Update Publications → Run workflow**). Si hay cambios, hace commit del JSON
actualizado y el sitio se redepliega.

### Secreto requerido: `SERPAPI_API_KEY`

Google Scholar **bloquea el scraping directo desde las IPs de GitHub Actions**, así que la
sincronización con Scholar usa la API de [SerpApi](https://serpapi.com) (su plan gratuito de
100 búsquedas/mes alcanza de sobra para la corrida semanal).

Para que funcione en CI debe existir el secreto de repositorio **`SERPAPI_API_KEY`**:

1. **Settings → Secrets and variables → Actions → New repository secret**
2. Name: `SERPAPI_API_KEY` — Secret: la API key de SerpApi.

El secreto queda cifrado y nunca se sube al repositorio. Si falta o falla, el script **no se cae**:
registra una advertencia, conserva las publicaciones de Scholar ya sincronizadas y actualiza el
resto con OpenAlex (así nunca se pierden publicaciones).

### Correr la sincronización localmente

```bash
SERPAPI_API_KEY=tu_clave npm run sync:publications
```

Sin la variable, el script intenta el scraping público de Scholar como último recurso
(poco fiable fuera de una IP residencial).

## Variables de entorno del script

| Variable                   | Por defecto            | Descripción                                   |
| -------------------------- | ---------------------- | --------------------------------------------- |
| `SERPAPI_API_KEY`          | —                      | Clave de SerpApi para sincronizar Scholar.    |
| `OPENALEX_AUTHOR_ID`       | `A5081714132`          | Autor de OpenAlex a consultar.                |
| `GOOGLE_SCHOLAR_AUTHOR_ID` | `R7gjbGIAAAAJ`         | Perfil de Google Scholar a consultar.         |
| `OPENALEX_MAILTO`          | —                      | Email para el "polite pool" de OpenAlex.      |
| `OPENALEX_API_KEY`         | —                      | API key opcional de OpenAlex.                 |
