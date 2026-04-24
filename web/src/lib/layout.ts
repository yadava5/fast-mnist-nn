/**
 * Compute node positions for a rectangular grid laid out on the X/Y plane
 * at a fixed Z depth. Used by the 3D neural-net hero to place layer nodes.
 */
export function gridPositions(
  cols: number,
  rows: number,
  width: number,
  height: number,
  z: number,
): [number, number, number][] {
  const positions: [number, number, number][] = [];
  const xStep = cols > 1 ? width / (cols - 1) : 0;
  const yStep = rows > 1 ? height / (rows - 1) : 0;
  const x0 = -width / 2;
  const y0 = -height / 2;
  for (let r = 0; r < rows; r += 1) {
    for (let c = 0; c < cols; c += 1) {
      positions.push([x0 + c * xStep, y0 + r * yStep, z]);
    }
  }
  return positions;
}
