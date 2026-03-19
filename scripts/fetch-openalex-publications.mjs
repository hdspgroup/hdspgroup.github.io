import { mkdir, writeFile } from 'node:fs/promises';
import path from 'node:path';
import process from 'node:process';

const AUTHOR_ID = process.env.OPENALEX_AUTHOR_ID ?? 'A5081714132';
const OUTPUT_PATH = process.env.OPENALEX_OUTPUT_PATH ?? 'src/data/openalex-publications.json';
const PER_PAGE = Number(process.env.OPENALEX_PER_PAGE ?? 200);
const MAILTO = process.env.OPENALEX_MAILTO ?? '';
const API_KEY = process.env.OPENALEX_API_KEY ?? '';
const USER_AGENT = process.env.OPENALEX_USER_AGENT ?? 'hdspgroup-publications-sync/1.0';

function buildUrl(base, params = {}) {
  const url = new URL(base);

  for (const [key, value] of Object.entries(params)) {
    if (value !== undefined && value !== null && value !== '') {
      url.searchParams.set(key, String(value));
    }
  }

  return url;
}

async function fetchJson(url) {
  const response = await fetch(url, {
    headers: {
      'User-Agent': USER_AGENT,
    },
  });

  if (!response.ok) {
    const body = await response.text();
    throw new Error(`OpenAlex request failed (${response.status} ${response.statusText}): ${body.slice(0, 500)}`);
  }

  return response.json();
}

function normalizeDoi(doiUrl) {
  if (!doiUrl) return null;
  return doiUrl.replace(/^https?:\/\/doi\.org\//i, '');
}

function normalizeLocation(location) {
  if (!location) return null;

  return {
    landingPageUrl: location.landing_page_url ?? null,
    pdfUrl: location.pdf_url ?? null,
    source: location.source
      ? {
          displayName: location.source.display_name ?? null,
          type: location.source.type ?? null,
        }
      : null,
  };
}

function normalizeWork(work) {
  const bestLocation = normalizeLocation(work.best_oa_location);
  const primaryLocation = normalizeLocation(work.primary_location);
  const authors = (work.authorships ?? [])
    .map((authorship) => authorship.author?.display_name)
    .filter(Boolean);

  return {
    id: work.id,
    title: work.display_name ?? work.title ?? 'Untitled work',
    publicationDate: work.publication_date ?? null,
    publicationYear: work.publication_year ?? null,
    citedByCount: work.cited_by_count ?? 0,
    doi: normalizeDoi(work.doi),
    doiUrl: work.doi ?? null,
    type: work.type ?? null,
    typeCrossref: work.type_crossref ?? null,
    isOpenAccess: Boolean(work.open_access?.is_oa),
    authors,
    authorsCount: authors.length,
    primaryLocation,
    bestOaLocation: bestLocation,
    venue:
      bestLocation?.source?.displayName ??
      primaryLocation?.source?.displayName ??
      work.primary_location?.source?.display_name ??
      null,
    venueType:
      bestLocation?.source?.type ??
      primaryLocation?.source?.type ??
      work.primary_location?.source?.type ??
      null,
    topics: (work.topics ?? []).map((topic) => topic.display_name).filter(Boolean).slice(0, 5),
    openAlexUrl: work.id ?? null,
  };
}

function normalizeAuthor(author) {
  return {
    id: author.id,
    name: author.display_name ?? 'Unknown author',
    worksCount: author.works_count ?? 0,
    citedByCount: author.cited_by_count ?? 0,
    openAlexUrl: author.id ?? null,
    orcid: author.orcid ?? author.ids?.orcid ?? null,
    institutions: (author.last_known_institutions ?? [])
      .map((institution) => institution.display_name)
      .filter(Boolean),
  };
}

async function fetchAuthorProfile() {
  const url = buildUrl(`https://api.openalex.org/authors/${AUTHOR_ID}`, {
    mailto: MAILTO,
    api_key: API_KEY,
  });

  const author = await fetchJson(url);
  return normalizeAuthor(author);
}

async function fetchAllWorks() {
  const works = [];
  let cursor = '*';

  while (cursor) {
    const url = buildUrl('https://api.openalex.org/works', {
      filter: `authorships.author.id:${AUTHOR_ID},is_paratext:false`,
      sort: 'publication_date:desc',
      cursor,
      'per-page': PER_PAGE,
      mailto: MAILTO,
      api_key: API_KEY,
    });

    const payload = await fetchJson(url);
    const batch = (payload.results ?? []).map(normalizeWork);
    works.push(...batch);

    cursor = payload.meta?.next_cursor ?? null;

    if (!batch.length) {
      break;
    }
  }

  const deduped = Array.from(new Map(works.map((work) => [work.id, work])).values());

  deduped.sort((a, b) => {
    const dateA = a.publicationDate ?? '';
    const dateB = b.publicationDate ?? '';
    return dateA < dateB ? 1 : dateA > dateB ? -1 : 0;
  });

  return deduped;
}

async function main() {
  const [profile, publications] = await Promise.all([fetchAuthorProfile(), fetchAllWorks()]);

  const data = {
    source: 'OpenAlex',
    authorId: AUTHOR_ID,
    fetchedAt: new Date().toISOString(),
    profile,
    publications,
  };

  const absoluteOutputPath = path.resolve(process.cwd(), OUTPUT_PATH);
  await mkdir(path.dirname(absoluteOutputPath), { recursive: true });
  await writeFile(absoluteOutputPath, `${JSON.stringify(data, null, 2)}\n`, 'utf8');

  console.log(`Saved ${publications.length} publications to ${OUTPUT_PATH}`);
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
