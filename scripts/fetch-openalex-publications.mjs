import { mkdir, writeFile } from 'node:fs/promises';
import path from 'node:path';
import process from 'node:process';

const AUTHOR_ID = process.env.OPENALEX_AUTHOR_ID ?? 'A5081714132';
const OUTPUT_PATH = process.env.OPENALEX_OUTPUT_PATH ?? 'src/data/openalex-publications.json';
const PER_PAGE = Number(process.env.OPENALEX_PER_PAGE ?? 200);
const DOI_CONCURRENCY = Number(process.env.DOI_CONCURRENCY ?? 4);
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

function normalizeDoi(doiValue) {
  if (!doiValue) return null;
  return String(doiValue).replace(/^https?:\/\/doi\.org\//i, '').trim() || null;
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

async function main() {
  const [profile, works] = await Promise.all([fetchAuthorProfile(), fetchAllOpenAlexWorks()]);

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

  const cleanedPublications = publications.filter(Boolean).sort((a, b) => {
    const dateA = a.publicationDate ?? '';
    const dateB = b.publicationDate ?? '';
    return dateA < dateB ? 1 : dateA > dateB ? -1 : 0;
  });

  const data = {
    source: 'DOI.org metadata with OpenAlex DOI discovery',
    authorId: AUTHOR_ID,
    fetchedAt: new Date().toISOString(),
    profile,
    totals: {
      openAlexWorks: works.length,
      worksWithDoi: worksWithDoi.length,
      doiBackedWorks: cleanedPublications.length,
      omittedWithoutDoiCount: worksWithoutDoiCount,
      omittedUnavailableMetadataCount: worksWithDoi.length - cleanedPublications.length,
    },
    publications: cleanedPublications,
  };

  const absoluteOutputPath = path.resolve(process.cwd(), OUTPUT_PATH);
  await mkdir(path.dirname(absoluteOutputPath), { recursive: true });
  await writeFile(absoluteOutputPath, `${JSON.stringify(data, null, 2)}\n`, 'utf8');

  console.log(`Saved ${cleanedPublications.length} DOI-backed publications to ${OUTPUT_PATH}`);
  console.log(`Omitted works without DOI: ${worksWithoutDoiCount}`);
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
