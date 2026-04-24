/** @type {import('@commitlint/types').UserConfig} */
export default {
  extends: ['@commitlint/config-conventional'],
  rules: {
    // Subject: cap at 80 to leave room for GitHub's auto-appended " (#N)"
    // suffix on squash merges. Hand-written subjects still target ~66 chars
    // to stay comfortably under the cap after squashing.
    'header-max-length': [2, 'always', 80],

    // Disable body/footer line caps — we write long URLs in VALIDATION
    // sections and the default 100-char limit fails on real URLs.
    'body-max-line-length': [0],
    'footer-max-line-length': [0],

    // Blank-line rules: warn, don't block. Squash merges can collapse
    // the leading blank line between body and footer; not worth failing
    'body-leading-blank': [1, 'always'],
    'footer-leading-blank': [1, 'always'],

    // Case rules disabled: we often use acronyms in subjects ("ADR-0001",
    // "SIMD", "OKLCH", "MLP", "WASM") which config-conventional flags as
    // pascal/upper-case. Not worth the noise.
    'subject-case': [0],

    // Type must be one of these.
    'type-enum': [
      2,
      'always',
      ['feat', 'fix', 'perf', 'chore', 'docs', 'refactor', 'test', 'build', 'ci', 'style'],
    ],

    // Scope is a soft suggestion (level 1 = warning). Expanded to cover
    // the organic set we've used so far so existing commits are silent.
    'scope-enum': [
      1,
      'always',
      [
        // tracks
        'frontend',
        'backend',
        'optimization',
        'hygiene',
        // areas
        'docs',
        'viz',
        'canvas',
        'web',
        // infra
        'ci',
        'deps',
        'release',
        'github',
        'automation',
        'security',
        'meta',
        'hooks',
        'deploy',
        'theme',
        'utils',
        'components',
        'scaffolds',
      ],
    ],
  },
};
