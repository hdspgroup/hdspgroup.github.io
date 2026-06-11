import { getPermalink, getBlogPermalink, getAsset } from './utils/permalinks';

export const headerData = {
  links: [
    {
      text: 'About us',
      href: getPermalink('/'),
      icon: 'tabler:sparkles',
    },
    {
      text: 'People',
      href: getPermalink('/people'),
      icon: 'tabler:users-group',
    },
    {
      text: 'Publications',
      href: getPermalink('/publications'),
      icon: 'tabler:file-analytics',
    },
    {
      text: 'Contact',
      href: getPermalink('/contact'),
      icon: 'tabler:map-pin',
    },
    {
      text: 'Events',
      href: getBlogPermalink(),
      icon: 'tabler:calendar-event',
    },
    {
      text: 'DIRECTOR App',
      href: getPermalink('/homes/director-app'),
      disabled: true,
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
    Source code on <a class="text-primary underline" href="https://github.com/hdspgroup">GitHub</a> · All rights reserved.
  `,
};
