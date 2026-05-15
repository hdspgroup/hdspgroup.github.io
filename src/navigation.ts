import { getPermalink, getBlogPermalink, getAsset } from './utils/permalinks';

export const headerData = {
  links: [
    {
      text: 'Homes',
      links: [
        {
          text: 'DIRECTOR App',
          href: getPermalink('/homes/director-app'),
        },
      ],
    },
    {
      text: 'About us',
      href: getPermalink('/'),
    },
    {
      text: 'People',
      href: getPermalink('/people'),
    },
    {
      text: 'Publications',
      href: getPermalink('/publications'),
    },
    {
      text: 'Contact',
      href: getPermalink('/contact'),
    },
    {
      text: 'Events',
      href: getBlogPermalink(),
    },
  ],
  actions: [
    {
      text: 'Scholar',
      href: 'https://scholar.google.com/citations?user=R7gjbGIAAAAJ&hl=en',
      target: '_blank',
    },
  ],
};

export const footerData = {
  links: [],
  secondaryLinks: [],
  socialLinks: [
    { ariaLabel: 'Instagram', icon: 'tabler:brand-instagram', href: 'https://www.instagram.com/hdspgroup/' },
    { ariaLabel: 'RSS', icon: 'tabler:rss', href: getAsset('/rss.xml') },
    { ariaLabel: 'Github', icon: 'tabler:brand-github', href: 'https://github.com/hdspgroup' },
  ],
  footNote: `
    Source code on <a class="text-blue-600 underline dark:text-muted" href="https://github.com/hdspgroup">GitHub</a> · All rights reserved.
  `,
};
