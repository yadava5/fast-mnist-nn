/** @type {import('@commitlint/types').UserConfig} */
export default {
  extends: ['@commitlint/config-conventional'],
  rules: {
    'header-max-length': [2, 'always', 72],
    'type-enum': [
      2,
      'always',
      ['feat', 'fix', 'perf', 'chore', 'docs', 'refactor', 'test', 'build', 'ci', 'style'],
    ],
    'scope-enum': [
      1,
      'always',
      [
        'frontend',
        'backend',
        'optimization',
        'hygiene',
        'docs',
        'viz',
        'canvas',
        'ci',
        'deps',
        'release',
      ],
    ],
  },
};
