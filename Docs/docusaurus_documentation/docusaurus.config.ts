import {themes as prismThemes} from 'prism-react-renderer';
import remarkMath from 'remark-math';
import rehypeKatex from 'rehype-katex';
import type {Config} from '@docusaurus/types';
import type * as Preset from '@docusaurus/preset-classic';

const config: Config = {
  title: 'AMReX',
  tagline: 'Block-structured AMR software framework',
  favicon: 'img/favicon.ico',

  // Set the production url of your site here
  url: 'https://amrex-codes.github.io',
  // Set the /<baseUrl>/ pathname under which your site is served
  // For GitHub pages deployment, it is often '/<projectName>/'
  baseUrl: '/amrex/',

  // GitHub pages deployment config.
  // If you aren't using GitHub pages, you don't need these.
  organizationName: 'AMReX-Codes',
  projectName: 'amrex',

  onBrokenLinks: 'throw',
  onBrokenMarkdownLinks: 'warn',

  // Even if you don't use internationalization, you can use this field to set
  // useful metadata like html lang. For example, if your site is Chinese, you
  // may want to replace "en" with "zh-Hans".
  i18n: {
    defaultLocale: 'en',
    locales: ['en'],
  },
  markdown: {
    format: 'detect',
  },

  presets: [
    [
      'classic',
      {
        docs: {
          sidebarPath: './sidebars.ts',
          editUrl:
            'https://github.com/AMReX-Codes/amrex/tree/development/Docs/docusaurus_documentation/',
          remarkPlugins: [remarkMath],
          rehypePlugins: [rehypeKatex],
        },
        blog: false,
        theme: {
          customCss: './src/css/custom.css',
        },
      } satisfies Preset.Options,
    ],
  ],

  themeConfig: {
    // Replace with your project's social card
    image: 'img/docusaurus-social-card.jpg',
    navbar: {
      title: 'AMReX',
      logo: {
        alt: 'AMReX Logo',
        src: 'img/logo.svg',
      },
      items: [
        {
          type: 'docSidebar',
          sidebarId: 'tutorialSidebar',
          position: 'left',
          label: 'Docs',
        },
        {
          href: 'https://amrex-codes.github.io/amrex/doxygen/index.html',
          label: 'Doxygen',
          position: 'right',
        },
        {
          href: 'https://amrex-codes.github.io/amrex/tutorials_html/',
          label: 'Tutorials',
          position: 'right',
        },
        {
          href: 'https://github.com/AMReX-Codes/amrex',
          label: 'GitHub',
          position: 'right',
        },
      ],
    },
    footer: {
      style: 'dark',
      links: [
        {
          title: 'Docs',
          items: [
            {
              label: 'AMReX Documentation',
              to: '/docs/intro',
            },
            {
              label: 'Guided Tutorials',
              href: 'https://amrex-codes.github.io/amrex/tutorials_html/',
            },
          ],
        },
        {
          title: 'Community',
          items: [
            {
              label: 'Discussions',
              href: 'https://github.com/AMReX-Codes/amrex/discussions',
            },
            {
              label: 'Issues',
              href: 'https://github.com/AMReX-Codes/amrex/issues',
            },
          ],
        },
        {
          title: 'More',
          items: [
            {
              label: 'GitHub',
              href: 'https://github.com/AMReX-Codes/amrex',
            },
            {
              label: 'High Performance Software Foundation',
              href: 'https://hpsf.io/',
            },
          ],
        },
      ],
      copyright: `Copyright © ${new Date().getFullYear()} AMReX Team. Built with Docusaurus.`,
    },
    prism: {
      theme: prismThemes.github,
      darkTheme: prismThemes.dracula,
    },
  } satisfies Preset.ThemeConfig,
};

export default config;
