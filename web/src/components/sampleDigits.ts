import type { Stroke, StrokePoint } from './strokeReducer';

const DEFAULT_PRESSURE = 0.62;

function point(x: number, y: number, index: number): StrokePoint {
  return {
    x,
    y,
    pressure: DEFAULT_PRESSURE,
    t: index * 16,
  };
}

function stroke(id: string, coords: readonly (readonly [number, number])[]): Stroke {
  return {
    id,
    pointerType: 'mouse',
    points: coords.map(([x, y], index) => point(x, y, index)),
  };
}

export function createSampleDigitFive(): Stroke[] {
  const stamp = Date.now().toString(36);

  return [
    stroke(`sample-five-top-${stamp}`, [
      [207, 58],
      [177, 55],
      [143, 55],
      [108, 58],
      [83, 65],
    ]),
    stroke(`sample-five-spine-${stamp}`, [
      [86, 66],
      [80, 92],
      [78, 119],
      [84, 139],
      [111, 142],
      [144, 141],
      [177, 146],
    ]),
    stroke(`sample-five-bowl-${stamp}`, [
      [177, 146],
      [202, 154],
      [216, 175],
      [214, 199],
      [201, 221],
      [174, 235],
      [139, 235],
      [105, 226],
      [82, 209],
    ]),
  ];
}
