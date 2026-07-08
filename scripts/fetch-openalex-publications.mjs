import { mkdir, readFile, writeFile } from 'node:fs/promises';
import path from 'node:path';
import process from 'node:process';

const AUTHOR_ID = process.env.OPENALEX_AUTHOR_ID ?? 'A5081714132';
const OUTPUT_PATH = process.env.OPENALEX_OUTPUT_PATH ?? 'src/data/openalex-publications.json';
const PER_PAGE = Number(process.env.OPENALEX_PER_PAGE ?? 200);
const DOI_CONCURRENCY = Number(process.env.DOI_CONCURRENCY ?? 4);
const MAILTO = process.env.OPENALEX_MAILTO ?? '';
const API_KEY = process.env.OPENALEX_API_KEY ?? '';
const USER_AGENT = process.env.OPENALEX_USER_AGENT ?? 'hdspgroup-publications-sync/1.0';
const SCHOLAR_AUTHOR_ID = process.env.GOOGLE_SCHOLAR_AUTHOR_ID ?? 'R7gjbGIAAAAJ';
const SERPAPI_API_KEY = process.env.SERPAPI_API_KEY ?? '';
const SCHOLAR_PAGE_SIZE = Math.min(Number(process.env.GOOGLE_SCHOLAR_PAGE_SIZE ?? 100), 100);
const SCHOLAR_MAX_ARTICLES = Number(process.env.GOOGLE_SCHOLAR_MAX_ARTICLES ?? 2000);
const SCHOLAR_PAGE_DELAY_MS = Number(process.env.GOOGLE_SCHOLAR_PAGE_DELAY_MS ?? 750);

function buildUrl(base, params = {}) {
  const url = new URL(base);

  for (const [key, value] of Object.entries(params)) {
    if (value !== undefined && value !== null && value !== '') {
      url.searchParams.set(key, String(value));
    }
  }

  return url;
}

const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

async function fetchJson(url, headers = {}, attempt = 0) {
  const response = await fetch(url, {
    headers: {
      'User-Agent': USER_AGENT,
      ...headers,
    },
  });

  if (response.status === 429 && attempt < 3) {
    await sleep(1000 * (attempt + 1));
    return fetchJson(url, headers, attempt + 1);
  }

  if (!response.ok) {
    const body = await response.text();
    throw new Error(`Request failed (${response.status} ${response.statusText}): ${body.slice(0, 500)}`);
  }

  return response.json();
}

async function fetchText(url, headers = {}, attempt = 0) {
  const response = await fetch(url, {
    headers: {
      'User-Agent': USER_AGENT,
      ...headers,
    },
  });

  if (response.status === 429 && attempt < 3) {
    await sleep(1000 * (attempt + 1));
    return fetchText(url, headers, attempt + 1);
  }

  if (!response.ok) {
    const body = await response.text();
    throw new Error(`Request failed (${response.status} ${response.statusText}): ${body.slice(0, 500)}`);
  }

  return response.text();
}

function normalizeDoi(doiValue) {
  if (!doiValue) return null;
  return String(doiValue).replace(/^https?:\/\/doi\.org\//i, '').trim() || null;
}

function decodeHtmlEntities(value) {
  return String(value ?? '')
    .replace(/&#(\d+);/g, (_, code) => String.fromCharCode(Number(code)))
    .replace(/&#x([a-f0-9]+);/gi, (_, code) => String.fromCharCode(Number.parseInt(code, 16)))
    .replace(/&amp;/g, '&')
    .replace(/&lt;/g, '<')
    .replace(/&gt;/g, '>')
    .replace(/&quot;/g, '"')
    .replace(/&#39;/g, "'")
    .replace(/&nbsp;/g, ' ');
}

function htmlToText(value) {
  return decodeHtmlEntities(String(value ?? '').replace(/<[^>]*>/g, ' '))
    .replace(/\s+/g, ' ')
    .trim();
}

function normalizeText(value) {
  return String(value ?? '')
    .normalize('NFKD')
    .replace(/[\u0300-\u036f]/g, '')
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, ' ')
    .trim();
}

function normalizeTitle(value) {
  if (Array.isArray(value)) return value.find(Boolean) ?? null;
  return value ?? null;
}

function normalizeVenue(value) {
  if (Array.isArray(value)) return value.find(Boolean) ?? null;
  return value ?? null;
}

function normalizeArguello(value) {
  return typeof value === 'string' ? value.replaceAll('Argüello', 'Arguello') : value;
}

function splitScholarAuthors(value) {
  return String(value ?? '')
    .split(/\s*,\s*/)
    .map((author) => normalizeArguello(author.trim()))
    .filter(Boolean);
}

function parseScholarYear(article) {
  const directYear = Number(article?.year);
  if (Number.isInteger(directYear)) return directYear;

  const match = String(article?.publication ?? '').match(/\b(19|20)\d{2}\b/g);
  return match?.length ? Number(match[match.length - 1]) : null;
}

function cleanScholarPublication(value, year) {
  const publication = String(value ?? '').trim();
  if (!publication) return null;

  return publication
    .replace(new RegExp(`\\s*,?\\s*${year}\\s*$`), '')
    .replace(/\s+/g, ' ')
    .trim();
}

function scholarUrl(pathname) {
  if (!pathname) return null;
  if (pathname.startsWith('http://') || pathname.startsWith('https://')) return pathname;
  return `https://scholar.google.com${pathname.startsWith('/') ? '' : '/'}${pathname}`;
}

function formatAuthor(author) {
  if (!author) return null;
  if (author.literal) return normalizeArguello(author.literal);
  return normalizeArguello([author.given, author.family].filter(Boolean).join(' ').trim() || null);
}

function extractDateParts(record) {
  const parts = record?.['date-parts']?.[0];
  if (!Array.isArray(parts) || !parts.length || !parts[0]) return null;

  const year = Number(parts[0]);
  const month = Number(parts[1] ?? 1);
  const day = Number(parts[2] ?? 1);

  return {
    year,
    date: `${String(year).padStart(4, '0')}-${String(month).padStart(2, '0')}-${String(day).padStart(2, '0')}`,
  };
}

function extractPublicationDate(metadata) {
  const candidates = [
    metadata.issued,
    metadata.published,
    metadata['published-print'],
    metadata['published-online'],
    metadata.created,
  ];

  for (const candidate of candidates) {
    const parsed = extractDateParts(candidate);
    if (parsed) return parsed;
  }

  return null;
}

function extractPdfUrl(metadata) {
  const links = Array.isArray(metadata.link) ? metadata.link : [];
  const pdfLike = links.find(
    (link) =>
      typeof link.URL === 'string' &&
      (link.URL.toLowerCase().includes('.pdf') || String(link['content-type'] ?? '').toLowerCase().includes('pdf'))
  );
  return pdfLike?.URL ?? null;
}

function extractLandingPageUrl(metadata, doi) {
  return metadata.URL ?? metadata.resource?.primary?.URL ?? (doi ? `https://doi.org/${doi}` : null);
}

function normalizeAuthorProfile(author) {
  return {
    id: author.id,
    name: normalizeArguello(author.display_name ?? 'Unknown author'),
    worksCount: author.works_count ?? 0,
    citedByCount: author.cited_by_count ?? 0,
    openAlexUrl: author.id ?? null,
    orcid: author.orcid ?? author.ids?.orcid ?? null,
    institutions: (author.last_known_institutions ?? [])
      .map((institution) => institution.display_name)
      .filter(Boolean),
  };
}

function normalizePublication(openalexWork, doiMetadata) {
  const doi = normalizeDoi(doiMetadata.DOI) ?? normalizeDoi(openalexWork.doi);
  const authors =
    (doiMetadata.author ?? [])
      .map(formatAuthor)
      .filter(Boolean) ||
    [];
  const publicationDate = extractPublicationDate(doiMetadata);
  const landingPageUrl = extractLandingPageUrl(doiMetadata, doi);
  const pdfUrl = extractPdfUrl(doiMetadata);
  const venue = normalizeVenue(doiMetadata['container-title']) ?? doiMetadata.publisher ?? null;

  return {
    id: openalexWork.id ?? (doi ? `https://doi.org/${doi}` : null),
    title: normalizeTitle(doiMetadata.title) ?? openalexWork.display_name ?? openalexWork.title ?? 'Untitled work',
    publicationDate: publicationDate?.date ?? openalexWork.publication_date ?? null,
    publicationYear: publicationDate?.year ?? openalexWork.publication_year ?? null,
    doi,
    doiUrl: doi ? `https://doi.org/${doi}` : null,
    type: doiMetadata.type ?? openalexWork.type ?? null,
    typeCrossref: doiMetadata.type ?? openalexWork.type_crossref ?? null,
    authors,
    authorsCount: authors.length,
    primaryLocation: {
      landingPageUrl,
      pdfUrl,
    },
    bestOaLocation: pdfUrl
      ? {
          landingPageUrl,
          pdfUrl,
        }
      : null,
    venue,
    venueType: doiMetadata.type ?? null,
    topics: (Array.isArray(doiMetadata.subject) ? doiMetadata.subject : []).filter(Boolean).slice(0, 5),
  };
}

function normalizeScholarPublication(article) {
  const year = parseScholarYear(article);
  const title = normalizeTitle(article?.title) ?? 'Untitled work';
  const authors = splitScholarAuthors(article?.authors);
  const landingPageUrl = article?.link ?? null;

  return {
    id: article?.citation_id ? `google-scholar:${article.citation_id}` : `google-scholar:${normalizeText(title)}`,
    title,
    publicationDate: null,
    publicationYear: year,
    doi: null,
    doiUrl: null,
    type: 'scholar-article',
    typeCrossref: null,
    authors,
    authorsCount: authors.length,
    primaryLocation: {
      landingPageUrl,
      pdfUrl: null,
    },
    bestOaLocation: null,
    venue: cleanScholarPublication(article?.publication, year),
    venueType: null,
    topics: [],
    source: 'google_scholar',
  };
}

function parseScholarProfileHtml(html) {
  if (/unusual traffic|recaptcha|\/sorry\//i.test(html)) {
    throw new Error('Google Scholar returned a verification page instead of the author profile.');
  }

  return Array.from(html.matchAll(/<tr class="gsc_a_tr">([\s\S]*?)<\/tr>/g))
    .map(([, row]) => {
      const titleMatch = row.match(
        /<a\b(?=[^>]*\bclass="[^"]*\bgsc_a_at\b[^"]*")(?=[^>]*\bhref="([^"]+)")[^>]*>([\s\S]*?)<\/a>/
      );
      if (!titleMatch) return null;

      const href = decodeHtmlEntities(titleMatch[1]);
      const title = htmlToText(titleMatch[2]);
      const details = Array.from(row.matchAll(/<div class="gs_gray">([\s\S]*?)<\/div>/g)).map((match) =>
        htmlToText(match[1])
      );
      const year = row.match(/<span[^>]*class="[^"]*\bgsc_a_hc\b[^"]*"[^>]*>(\d{4})<\/span>/)?.[1] ?? null;
      const citationId = href.match(/[?&]citation_for_view=([^&"]+)/)?.[1] ?? null;

      return {
        title,
        link: scholarUrl(href),
        citation_id: citationId ? decodeHtmlEntities(citationId) : null,
        authors: details[0] ?? '',
        publication: details[1] ?? '',
        year,
      };
    })
    .filter(Boolean);
}

async function fetchScholarPublicProfileArticles() {
  const articles = [];

  for (let cstart = 0; cstart < SCHOLAR_MAX_ARTICLES; cstart += SCHOLAR_PAGE_SIZE) {
    const pageSize = Math.min(SCHOLAR_PAGE_SIZE, SCHOLAR_MAX_ARTICLES - cstart);
    const url = buildUrl('https://scholar.google.com/citations', {
      user: SCHOLAR_AUTHOR_ID,
      hl: 'en',
      cstart,
      pagesize: pageSize,
      sortby: 'pubdate',
    });

    const html = await fetchText(url, {
      'User-Agent': 'Mozilla/5.0 hdspgroup-publications-sync/1.0',
      Accept: 'text/html,application/xhtml+xml',
    });
    const batch = parseScholarProfileHtml(html);
    articles.push(...batch);

    console.log(`Fetched ${batch.length} Google Scholar articles from offset ${cstart}.`);

    if (batch.length < pageSize) {
      break;
    }

    await sleep(SCHOLAR_PAGE_DELAY_MS);
  }

  return articles;
}

function publicationKeys(publication) {
  const keys = [];
  const doi = normalizeDoi(publication.doi);
  const title = normalizeText(publication.title);

  if (doi) keys.push(`doi:${doi.toLowerCase()}`);
  if (title) keys.push(`title:${title}`);

  return keys;
}

function mergePublicationDetails(existing, incoming) {
  const incomingYear = incoming.publicationYear ?? 0;
  const existingYear = existing.publicationYear ?? 0;
  const preferIncomingDetails = incoming.source === 'google_scholar' && incomingYear >= existingYear;
  const base = preferIncomingDetails ? incoming : existing;
  const supplemental = preferIncomingDetails ? existing : incoming;

  return {
    ...base,
    doi: base.doi ?? supplemental.doi ?? null,
    doiUrl: base.doiUrl ?? supplemental.doiUrl ?? null,
    bestOaLocation: base.bestOaLocation ?? supplemental.bestOaLocation ?? null,
    primaryLocation: {
      landingPageUrl: base.primaryLocation?.landingPageUrl ?? supplemental.primaryLocation?.landingPageUrl ?? null,
      pdfUrl: base.primaryLocation?.pdfUrl ?? supplemental.primaryLocation?.pdfUrl ?? null,
    },
    topics: base.topics?.length ? base.topics : supplemental.topics ?? [],
  };
}

function mergePublications(openAlexPublications, scholarPublications) {
  const merged = new Map();

  for (const publication of [...openAlexPublications, ...scholarPublications]) {
    const keys = publicationKeys(publication);
    const existing = keys.map((key) => merged.get(key)).find(Boolean);

    if (!existing) {
      keys.forEach((key) => merged.set(key, publication));
      continue;
    }

    const next = mergePublicationDetails(existing, publication);
    publicationKeys(next).forEach((key) => merged.set(key, next));
  }

  return Array.from(new Set(merged.values())).sort((a, b) => {
    const dateA = a.publicationDate ?? '';
    const dateB = b.publicationDate ?? '';
    return dateA < dateB ? 1 : dateA > dateB ? -1 : a.title.localeCompare(b.title);
  });
}

async function fetchAuthorProfile() {
  const url = buildUrl(`https://api.openalex.org/authors/${AUTHOR_ID}`, {
    mailto: MAILTO,
    api_key: API_KEY,
  });

  const author = await fetchJson(url);
  return normalizeAuthorProfile(author);
}

async function fetchAllOpenAlexWorks() {
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
    const batch = payload.results ?? [];
    works.push(...batch);

    cursor = payload.meta?.next_cursor ?? null;

    if (!batch.length) {
      break;
    }
  }

  return Array.from(new Map(works.map((work) => [work.id, work])).values());
}

async function fetchScholarViaSerpApi() {
  const articles = [];

  for (let start = 0; start < SCHOLAR_MAX_ARTICLES; start += SCHOLAR_PAGE_SIZE) {
    const num = Math.min(SCHOLAR_PAGE_SIZE, SCHOLAR_MAX_ARTICLES - start);
    const url = buildUrl('https://serpapi.com/search.json', {
      engine: 'google_scholar_author',
      author_id: SCHOLAR_AUTHOR_ID,
      hl: 'en',
      sort: 'pubdate',
      start,
      num,
      api_key: SERPAPI_API_KEY,
    });

    const payload = await fetchJson(url);

    if (payload.error) {
      throw new Error(`SerpApi Google Scholar request failed: ${payload.error}`);
    }

    const batch = payload.articles ?? [];
    articles.push(...batch);

    console.log(`Fetched ${batch.length} Google Scholar articles from SerpApi offset ${start}.`);

    if (batch.length < num || !payload.serpapi_pagination?.next) {
      break;
    }

    await sleep(SCHOLAR_PAGE_DELAY_MS);
  }

  return articles;
}

async function fetchScholarPublications() {
  if (SERPAPI_API_KEY) {
    const articles = await fetchScholarViaSerpApi();
    return articles.map(normalizeScholarPublication).filter((publication) => publication.publicationYear);
  }

  console.log('SERPAPI_API_KEY is not configured. Using Google Scholar public profile fallback.');
  const articles = await fetchScholarPublicProfileArticles();
  return articles.map(normalizeScholarPublication).filter((publication) => publication.publicationYear);
}

async function fetchDoiMetadata(doi) {
  const citationUrl = buildUrl('https://citation.doi.org/metadata', { doi });

  try {
    return await fetchJson(citationUrl, {
      Accept: 'application/json',
    });
  } catch (citationError) {
    const doiUrl = `https://doi.org/${encodeURIComponent(doi)}`;

    try {
      return await fetchJson(doiUrl, {
        Accept: 'application/vnd.citationstyles.csl+json',
      });
    } catch (doiError) {
      throw new Error(`${citationError.message}; fallback failed (${doiError.message})`);
    }
  }
}

async function mapInBatches(items, concurrency, mapper) {
  const results = [];

  for (let index = 0; index < items.length; index += concurrency) {
    const batch = items.slice(index, index + concurrency);
    const resolved = await Promise.all(batch.map(mapper));
    results.push(...resolved);
  }

  return results;
}

function isScholarPublication(publication) {
  return publication?.source === 'google_scholar' || String(publication?.id ?? '').startsWith('google-scholar:');
}

async function readExistingScholarPublications(outputPath) {
  try {
    const parsed = JSON.parse(await readFile(outputPath, 'utf8'));
    return (parsed.publications ?? []).filter(isScholarPublication);
  } catch {
    return [];
  }
}

async function main() {
  const absoluteOutputPath = path.resolve(process.cwd(), OUTPUT_PATH);

  const [profile, works, fetchedScholarPublications] = await Promise.all([
    fetchAuthorProfile(),
    fetchAllOpenAlexWorks(),
    fetchScholarPublications().catch((error) => {
      console.warn(`Google Scholar sync failed: ${error.message}`);
      return [];
    }),
  ]);

  // If the Scholar sync failed or returned nothing (e.g. Google blocked the CI
  // runner and no SERPAPI_API_KEY is configured), reuse the Scholar-only entries
  // already stored so we never drop publications that OpenAlex does not have.
  let scholarPublications = fetchedScholarPublications;
  if (!scholarPublications.length) {
    const preserved = await readExistingScholarPublications(absoluteOutputPath);
    if (preserved.length) {
      console.warn(`Preserving ${preserved.length} previously synced Google Scholar publications.`);
      scholarPublications = preserved;
    }
  }

  const worksWithDoi = works.filter((work) => normalizeDoi(work.doi));
  const worksWithoutDoiCount = works.length - worksWithDoi.length;

  const publications = await mapInBatches(worksWithDoi, DOI_CONCURRENCY, async (work) => {
    const doi = normalizeDoi(work.doi);

    try {
      const doiMetadata = await fetchDoiMetadata(doi);
      return normalizePublication(work, doiMetadata);
    } catch (error) {
      console.warn(`Skipping DOI ${doi}: ${error.message}`);
      return null;
    }
  });

  const doiBackedPublications = publications.filter(Boolean).sort((a, b) => {
    const dateA = a.publicationDate ?? '';
    const dateB = b.publicationDate ?? '';
    return dateA < dateB ? 1 : dateA > dateB ? -1 : 0;
  });
  const cleanedPublications = mergePublications(doiBackedPublications, scholarPublications);

  const data = {
    source: 'DOI.org metadata with OpenAlex DOI discovery, plus Google Scholar profile sync',
    authorId: AUTHOR_ID,
    scholarAuthorId: SCHOLAR_AUTHOR_ID,
    fetchedAt: new Date().toISOString(),
    profile,
    totals: {
      openAlexWorks: works.length,
      worksWithDoi: worksWithDoi.length,
      doiBackedWorks: doiBackedPublications.length,
      scholarWorks: scholarPublications.length,
      mergedWorks: cleanedPublications.length,
      omittedWithoutDoiCount: worksWithoutDoiCount,
      omittedUnavailableMetadataCount: worksWithDoi.length - doiBackedPublications.length,
    },
    publications: cleanedPublications,
  };

  await mkdir(path.dirname(absoluteOutputPath), { recursive: true });
  await writeFile(absoluteOutputPath, `${JSON.stringify(data, null, 2)}\n`, 'utf8');

  console.log(`Saved ${cleanedPublications.length} merged publications to ${OUTPUT_PATH}`);
  console.log(`Google Scholar publications merged: ${scholarPublications.length}`);
  console.log(`Omitted works without DOI: ${worksWithoutDoiCount}`);
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
